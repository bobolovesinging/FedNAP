"""Build all SHP roles together, starting from pinned upstream datasets."""
import argparse
import hashlib
import itertools
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from scripts.prepare_server_validation import (
    comparison_rank, file_sha256, prompt_id, read_records,
)
from scripts.prepare_shp_table_vii_data import prepare

SHP_REVISION = "e94b5f32602712d78ed494fe79105b1959396686"
SAFE_REVISION = "66465198fb8b9cbc6a7bdff124afaf35158d807c"
REFERENCE_SHA256 = "83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4"


def question(row):
    return prompt_id(row, "instruction")


def rank(key):
    return hashlib.sha256(key.encode("utf-8")).digest()


def groups(path, blocked):
    result = defaultdict(list)
    seen = set()
    for row in read_records(path):
        key = question(row)
        if key in blocked:
            continue
        answers = [row["output_A"], row["output_B"]]
        choice = int(row["choice"])
        pair = (key, answers[choice], answers[1 - choice])
        if pair not in seen:
            result[key].append(row)
            seen.add(pair)
    return result


def take(pool, n, cap, protected=(), *, smallest_first=False):
    """Select at most cap pairs per question and reserve the entire group."""
    available = pool.keys() - set(protected)
    if smallest_first:
        # Smaller groups leave fewer unused comparisons when the cap applies.
        order = sorted(available, key=lambda key: (len(pool[key]), rank(key)))
    else:
        order = sorted(available, key=rank)
    selected = []
    for key in order:
        rows = sorted(pool.pop(key), key=lambda row: comparison_rank(row, "instruction"))
        selected.extend(rows[:min(cap, n - len(selected))])
        if len(selected) == n:
            return selected
    raise ValueError(f"Only {len(selected)} eligible comparisons for {n} requested")


def prompt(text, domain, source):
    return {"instruction": text, "category": domain, "prompt_source": source,
            "prompt_id": hashlib.sha256(f"{source}\0{text}".encode()).hexdigest()}


def stage(raw, safe, evaluation, destination, head_train=24000, head_eval=1000,
          panel=256, client_num=200):
    """Write the new directory only after verifying the serialized role files."""
    if destination.exists():
        raise FileExistsError(destination)
    if file_sha256(evaluation) != REFERENCE_SHA256:
        raise ValueError("Unexpected AlpacaEval reference SHA-256")
    benchmark = {question(r) for r in read_records(evaluation)}
    val = groups(raw / "shp_validation.jsonl", benchmark)
    test = groups(raw / "shp_test.jsonl", benchmark)
    # Keep the DPO source pool independent of the internal holdout allocation.
    reddit_path = "prompts/reddit.json"
    reddit = {key: prompt(rows[0]["instruction"], rows[0]["domain"], "reddit")
              for pool in (val, test) for key, rows in sorted(pool.items())}
    train = groups(raw / "shp_train.jsonl", benchmark | set(reddit))
    # Keep normalized variants on the same client and protect each client
    # from becoming empty when its other questions go to head pretraining.
    reserved = {}
    for key in sorted(train, key=rank):
        client = train[key][0]["category"]
        reserved.setdefault(client, key)
        for row in train[key]:
            row["category"] = client
    if set(reserved) != {f"Client_{i}" for i in range(client_num)}:
        raise ValueError("An SHP client is empty after held-out question exclusion")

    protected = set(reserved.values())
    head_rows = take(train, head_train, 2, protected, smallest_first=True)
    roles = {"selector/shp_validation.jsonl": take(train, panel, 1, protected),
             "calibration.jsonl": take(train, panel, 1, protected),
             "heads/shp_validation.jsonl": take(train, head_eval, 2, protected),
             "heads/shp_test.jsonl": take(train, head_eval, 2, protected),
             "heads/shp_train.jsonl": head_rows}
    test_keys = sorted(train.keys() - protected, key=rank)
    selector_keys = test_keys[:min(512, len(test_keys) // 2)]
    roles["selector/shp_test.jsonl"] = list(itertools.chain.from_iterable(
        train.pop(key) for key in selector_keys))
    for name, split in (
        ("selector/shp_validation.jsonl", "validation"),
        ("calibration.jsonl", "validation"),
        ("heads/shp_validation.jsonl", "validation"),
        ("heads/shp_test.jsonl", "test"),
        ("selector/shp_test.jsonl", "test"),
    ):
        for row in roles[name]:
            row["category"] = f"Eval_{split}_{row['domain']}"

    roles["selector/shp_train.jsonl"] = itertools.chain.from_iterable(train.values())
    roles[reddit_path] = list(reddit.values())

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        out = Path(tmp) / "staged"
        out.mkdir()
        shutil.copyfile(evaluation, out / "alpacaeval.json")
        keys = {"alpacaeval.json": benchmark}
        records = {"alpacaeval.json": {"rows": sum(1 for _ in read_records(evaluation)),
                                       "groups": len(benchmark), "sha256": REFERENCE_SHA256}}

        def write(name, rows):
            path = out / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as f:
                if name.endswith(".json"):
                    json.dump(rows, f, ensure_ascii=False)
                else:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            unique = set()
            count = 0
            for row in read_records(path):
                unique.add(question(row))
                count += 1
            if not count:
                raise ValueError(f"Empty split: {name}")
            keys[name] = unique
            records[name] = {"rows": count, "groups": len(unique), "sha256": file_sha256(path)}

        for name, rows in roles.items():
            write(name, rows)
        blocked = set.union(*keys.values())
        safety = []
        for row in safe:
            item = prompt(row["prompt"], "saferlhf", "saferlhf")
            key = question(item)
            if key not in blocked:
                safety.append(item)
                blocked.add(key)
        write("prompts/saferlhf.json", safety)
        intersections = {f"{a} | {b}": len(keys[a] & keys[b])
                         for a, b in itertools.combinations(keys, 2)}
        for pair, overlap in intersections.items():
            if overlap:
                raise ValueError(f"Staged SHP roles share normalized questions: {pair}")
        source = json.loads((raw / "manifest.json").read_text(encoding="utf-8"))
        tokenizer = Path(source.pop("tokenizer_path"))
        source["tokenizer_sha256"] = {
            name: file_sha256(tokenizer / name)
            for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json")
        }
        manifest = {"shp_revision": SHP_REVISION, "saferlhf_revision": SAFE_REVISION,
                    "source": source,
                    "files": records, "intersections": intersections}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        out.rename(destination)
    return manifest


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if file_sha256(args.evaluation) != REFERENCE_SHA256:
        raise ValueError("Use the pinned AlpacaEval reference")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as tmp:
        raw = Path(tmp)
        prepare(raw, dataset_revision=SHP_REVISION, tokenizer_path=args.tokenizer_path)
        safe = load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt", revision=SAFE_REVISION, split="train")
        manifest = stage(raw, safe, args.evaluation, args.output_dir)
    print(json.dumps(manifest["files"], indent=2))


if __name__ == "__main__":
    main()
