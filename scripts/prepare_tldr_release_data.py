"""Download official TLDR files and prepare worker-preserving comparisons."""
import argparse
import importlib.util
import json
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import sleep
from urllib.error import URLError
from urllib.request import urlopen

from prepare_server_validation import file_sha256

SOURCE = 'https://openaipublic.blob.core.windows.net/summarize-from-feedback'


def download(url, target, expected):
    if target.exists():
        if file_sha256(target) != expected:
            raise ValueError(f'Checksum mismatch in cached input: {target}')
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        with NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
        try:
            with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
                while chunk := response.read(1024 * 1024):
                    stream.write(chunk)
            if file_sha256(temporary) != expected:
                raise ValueError(f'Official download failed checksum verification: {url}')
            temporary.replace(target)
            return
        except (TimeoutError, ConnectionError, URLError):
            if attempt == 2:
                raise
            sleep(2 ** attempt)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(root):
    root = Path(root)
    raw = root / 'reddit-tldr-comparison'
    output = root / 'tldr_full'
    if output.exists():
        raise FileExistsError('Use a fresh output root for prepared comparisons')
    root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=root) as tmp:
        staged = Path(tmp) / 'tldr_full'
        staged.mkdir()
        result = prepare_comparisons(raw, staged)
        staged.rename(output)
    return result


def load_tldr_label_rules():
    # Import the pure label implementation without initializing the FL runtime.
    code = Path(__file__).parents[1] / 'federatedscope/llm/dataset/tldr_preference.py'
    spec = importlib.util.spec_from_file_location('tldr_labels', code)
    labels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(labels)
    return labels


def prepare_comparisons(raw, output):
    checksums = Path(__file__).with_name('tldr_sources.sha256').read_text().splitlines()
    expected = {name: sha for sha, name in (line.split() for line in checksums)}
    targets = {key: output / f'tldr_cmpr_{key}_bias.jsonl'
               for key in ('train', 'val', 'test')}
    labels = load_tldr_label_rules()
    splits = {'train': 'train', 'valid1': 'val', 'valid2': 'test'}
    records = {key: [] for key in targets}
    source_hashes = {}
    for index in [*range(3, 21), 22]:
        path = raw / f'batch{index}.json'
        download(f'{SOURCE}/dataset/comparisons/{path.name}', path, expected[path.name])
        source_hashes[path.name] = file_sha256(path)
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                if item['split'] not in splits:
                    raise ValueError(f'Unknown split: {item["split"]}')
                info = item['info']
                row = {
                    'id': info['id'], 'post': info['post'], 'subreddit': info['subreddit'],
                    'title': info['title'], 'output_A': item['summaries'][0]['text'],
                    'output_B': item['summaries'][1]['text'],
                    'choice': int(item['choice']), 'category': item['worker'],
                    'split': item['split'],
                }
                if row['choice'] not in (0, 1):
                    raise ValueError('Comparison choice must be 0 or 1')
                row.update(labels.compute_causal_bias_labels(row))
                records[splits[item['split']]].append(row)
    # Keep official comparisons and human-summary pools as separate inputs.
    for source, destination in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        path = raw / f'reddit-tldr_{destination}.jsonl'
        download(f'{SOURCE}/datasets/tldr_3_filtered/{source}.jsonl', path, expected[path.name])
        source_hashes[path.name] = expected[path.name]
    for split, path in targets.items():
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in records[split]:
                stream.write(json.dumps(row) + '\n')
    manifest = {
        'source': SOURCE, 'source_sha256': source_hashes,
        'split_mapping': splits,
        'counts': {key: len(value) for key, value in records.items()},
        'training_workers': sorted({str(row['category']) for row in records['train']}),
        'files': {path.name: file_sha256(path)
                  for path in targets.values()},
    }
    (output / 'preparation.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='data')
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_root), indent=2))
