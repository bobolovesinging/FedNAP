"""Stage prompt-disjoint TLDR inputs for FedNAP from official comparison and summary files."""
import argparse
import csv
from collections import Counter
from itertools import combinations
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from prepare_server_validation import comparison_rank, file_sha256, normalized, read_records
from prepare_tldr_release_data import load_tldr_label_rules
from tldr_benchmark import TEST_SHA

HEAD_INDEX = Path(__file__).with_name('tldr_head_labels.tsv')
TRAIN_SHA = 'db75dd2169ffba8bc850b64965ec7c4124d38795ba15a6c265a5938348dc4e2a'
LEGACY_TRAIN_SHA = '01c4e50337ff74686807c8e6fdf82b543309f144e8a5d61dba72c9f112438bb4'
HUMAN = Path('reddit-tldr-comparison')
CMP = Path('tldr_full')


def post_key(row):
    value = normalized(row.get('data', row)['post'])
    if not value:
        raise ValueError('Missing source post')
    return hashlib.sha256(value.encode()).hexdigest()


class PostGroups:
    """Join matching Reddit IDs and normalized post texts across all sources."""
    def __init__(self):
        self.parent = {}
        self.by_id = {}

    def root(self, key):
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def add(self, row):
        key = post_key(row)
        self.parent.setdefault(key, key)
        reddit_id = row.get('id')
        if reddit_id:
            other = self.by_id.setdefault(str(reddit_id), key)
            left, right = sorted((self.root(key), self.root(other)))
            self.parent[right] = left

    def key(self, row):
        return self.root(post_key(row))


def ordered(keys):
    return sorted(set(keys), key=lambda k: hashlib.sha256(('42:' + k).encode()).digest())


def unique_comparisons(rows, groups):
    # A reversed display order is the same vote. Omit conflicting repeat votes.
    votes = {}
    for row in rows:
        a, b = (normalized(row[k]) for k in ('output_A', 'output_B'))
        if a == b:
            continue
        pair = (groups.key(row), row['category'], *sorted((a, b)))
        winner = (a, b)[row['choice']]
        if pair not in votes:
            votes[pair] = (winner, row)
        elif votes[pair][0] != winner:
            votes[pair] = (None, None)
    return [row for _, row in votes.values() if row is not None]


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        if path.suffix == '.json':
            json.dump(rows, stream, ensure_ascii=False)
            stream.write('\n')
        else:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def head_annotations(train):
    # The row indices refer to the pinned, worker-preserving official train order.
    if file_sha256(HEAD_INDEX) != '364deb7d7fac65e5dad613d2fed451ea0b9fae423d0fde58b1c8581dccb1a1b0':
        raise ValueError('Head annotation index checksum mismatch')
    if file_sha256(train) not in (TRAIN_SHA, LEGACY_TRAIN_SHA):
        raise ValueError('Official comparison order differs from the head-label index')
    with HEAD_INDEX.open(encoding='utf-8', newline='') as stream:
        labels = list(csv.DictReader(stream, delimiter='\t'))
    indices = [int(row.pop('train_row')) for row in labels]
    needed = set(indices)
    examples = {}
    for i, row in enumerate(read_records(train)):
        if i in needed:
            a, b = row['output_A'], row['output_B']
            if row['choice']:
                a, b = b, a
            examples[i] = dict(subreddit=row['subreddit'], post=row['post'],
                               chosen=a, rejected=b, choice=row['choice'])
    return [dict(data=examples[i], labels=label) for i, label in zip(indices, labels)]


def stage(raw_root, out):
    sources = {s: raw_root / CMP / f'tldr_cmpr_{s}_bias.jsonl'
               for s in ('train', 'val', 'test')}
    human = {s: raw_root / HUMAN / f'reddit-tldr_{s}.jsonl' for s in sources}
    preparation = json.loads((raw_root / CMP / 'preparation.json').read_text(encoding='utf-8'))
    for path in [*sources.values(), *human.values()]:
        expected = (preparation['files'] if path in sources.values()
                    else preparation['source_sha256'])[path.name]
        if file_sha256(path) != expected:
            raise ValueError(f'Prepared input checksum mismatch: {path}')
    if file_sha256(human['test']) != TEST_SHA:
        raise ValueError('Fixed test checksum mismatch')
    legacy_labels = preparation['files'][sources['train'].name] == LEGACY_TRAIN_SHA
    label_rules = load_tldr_label_rules() if legacy_labels else None
    groups = PostGroups()
    for path in [*sources.values(), *human.values()]:
        for row in read_records(path):
            if path in sources.values() and 'id' not in row:
                raise ValueError('Rebuild the official comparison files to retain Reddit IDs')
            groups.add(row)

    paths = {}

    def save(name, path, rows):
        if not rows:
            raise ValueError(f'Empty split: {name}')
        for row in rows:
            row['source_group'] = groups.key(row)
        paths[name] = path
        write_rows(out / path, rows)

    def save_comparisons(name, path, rows):
        if label_rules is not None:
            for row in rows:
                row.update(label_rules.compute_causal_bias_labels(row))
        save(name, path, rows)

    paths['benchmark'] = HUMAN / human['test'].name
    (out / HUMAN).mkdir()
    shutil.copyfile(human['test'], out / paths['benchmark'])
    benchmark_groups = {groups.key(r) for r in read_records(human['test'])}
    reserved = set(benchmark_groups)

    heads = [r for r in head_annotations(sources['train']) if groups.key(r) not in reserved]
    head_keys = ordered(groups.key(r) for r in heads)
    ntrain, nval = len(head_keys) * 8 // 10, len(head_keys) // 10
    membership = {k: ('train' if i < ntrain else 'val' if i < ntrain + nval else 'test')
                  for i, k in enumerate(head_keys)}
    for row in heads:
        row['split'] = membership[groups.key(row)]
    save('heads', Path('training/head_annotations.json'), heads)
    reserved.update(head_keys)

    comparisons = set()
    comparison_posts = set()
    for path in sources.values():
        for row in read_records(path):
            comparisons.add(groups.key(row))
            comparison_posts.add(row['post'])
    validation = unique_comparisons(read_records(sources['val']), groups)
    selected = {}
    for row in sorted(validation, key=lambda r: comparison_rank(r, 'post')):
        key = groups.key(row)
        if key not in reserved:
            selected.setdefault(key, row)
    rows = list(selected.values())
    if len(rows) < 512:
        raise ValueError('Need at least 512 eligible validation posts')
    for name, path, panel in [('calibration', Path('calibration.jsonl'), rows[:256]),
                              ('monitoring', CMP / 'tldr_cmpr_val_bias.jsonl', rows[256:512])]:
        save_comparisons(name, path, panel)
        reserved.update(groups.key(r) for r in panel)

    for split in ('test', 'train'):
        rows = unique_comparisons((r for r in read_records(sources[split])
                                   if groups.key(r) not in reserved), groups)
        save_comparisons('selector_' + split, CMP / sources[split].name, rows)
        reserved.update(groups.key(r) for r in rows)

    reserved.update(comparisons)

    def human_pool(records):
        rows = []
        seen = set()
        for row in records:
            key = groups.key(row)
            identity = (key, normalized(row['summary']))
            if key not in reserved and identity not in seen:
                seen.add(identity)
                row['summary'] = ' ' + row['summary'].lstrip()
                rows.append(row)
        return rows

    dev = human_pool(read_records(human['val']))
    keys = ordered(groups.key(r) for r in dev)
    val = set(keys[:len(keys) * 8 // 10])
    save('policy_val', HUMAN / 'policy_val.jsonl', [r for r in dev if groups.key(r) in val])
    save('policy_dev_holdout', HUMAN / 'policy_dev_holdout.jsonl',
         [r for r in dev if groups.key(r) not in val])
    reserved.update(keys)
    train_rows = list(read_records(human['train']))
    # Recover the original candidate slice in official row order.
    original_pool = [row for row in train_rows if row['post'] not in comparison_posts]
    original_candidates = original_pool[len(original_pool) * 3 // 5:]
    pool = human_pool(train_rows)
    available = {groups.key(row) for row in pool}

    candidates = {}
    for row in original_candidates:
        key = groups.key(row)
        if key in available:
            candidates.setdefault(key, row)
    save('sft', HUMAN / 'sft.jsonl',
         [row for row in pool if groups.key(row) not in candidates])
    save('candidates', HUMAN / 'candidates.jsonl',
         [{field: row[field] for field in ('id', 'subreddit', 'title', 'post')}
          for row in candidates.values()])

    # Audit the serialized outputs, including the three actual head partitions.
    assigned, counts, files = {}, Counter(), {}
    for name, path in paths.items():
        files[str(path.as_posix())] = file_sha256(out / path)
        for row in read_records(out / path):
            role = 'head_' + row['split'] if name == 'heads' else name
            counts[role] += 1
            key = groups.key(row)
            if name != 'benchmark' and row['source_group'] != key:
                raise ValueError(f'Invalid source group in {path}')
            assigned.setdefault(role, set()).add(key)
    overlaps = {f'{a}/{b}': len(assigned[a] & assigned[b]) for a, b in combinations(assigned, 2)}
    if any(overlaps.values()) or files[paths['benchmark'].as_posix()] != TEST_SHA:
        raise ValueError('Staged files failed the split audit')
    manifest = dict(rows=counts, groups={k: len(v) for k, v in assigned.items()},
                    intersections=overlaps, sha256=files,
                    candidate_allocation=dict(
                        original_prompt_rows=len(original_candidates),
                        retained_original_groups=len(candidates)),
                    inputs={p.name: file_sha256(p) for p in [*sources.values(), *human.values(), HEAD_INDEX]})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    target = args.output_dir
    if target.exists():
        parser.error('Choose a fresh staging directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as tmp:
        staged = Path(tmp) / 'staged'
        staged.mkdir()
        counts = stage(args.data_root, staged)
        staged.rename(target)
    print(json.dumps(counts, indent=2))
