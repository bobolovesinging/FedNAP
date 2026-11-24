#!/usr/bin/env python
"""Build preferences from validation-calibrated MAP-Elites margins."""

import argparse
import copy
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from federatedscope.llm.map_elites_improvement import (
    select_calibrated_ensemble,
)


def _load(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_dump(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _verified_load(path, expected_sha256):
    if _sha256(path) != expected_sha256:
        raise ValueError(f"margin cache hash mismatch: {path}")
    return _load(path)


def calibrate_preferences_file(pair_data, margin_manifest, output,
                               summary_output):
    pair_data = Path(pair_data)
    margin_manifest = Path(margin_manifest)
    output = Path(output)
    summary_output = Path(summary_output)
    cache = _load(margin_manifest)
    if not cache.get("complete") or len(cache.get("cells", [])) != 4:
        raise ValueError("margin manifest is incomplete")
    if _sha256(pair_data) != cache["pair_data_sha256"]:
        raise ValueError("pair data hash does not match margin cache")
    records = _load(pair_data)
    if int(cache["pair_count"]) > len(records):
        raise ValueError("margin cache exceeds available pair records")
    records = records[:int(cache["pair_count"])]
    labels = _verified_load(
        cache["validation_labels"], cache["validation_labels_sha256"])
    validation_margins = []
    pair_margins = []
    client_counts = []
    for cell in cache["cells"]:
        validation_path = cell["paths"]["validation_margins"]
        pair_path = cell["paths"]["pair_margins"]
        validation_margins.append(_verified_load(
            validation_path, cell["sha256"]["validation_margins"]))
        pair_margins.append(_verified_load(
            pair_path, cell["sha256"]["pair_margins"]))
        client_counts.append(int(cell["client_count"]))
    if any(len(column) != len(records) for column in pair_margins):
        raise ValueError("pair margin count does not match cached pair prefix")

    result = select_calibrated_ensemble(
        validation_margins, labels, pair_margins, client_counts)
    selected = []
    counts = Counter()
    for source, choice in zip(records, result["choices"]):
        sample = copy.deepcopy(source)
        sample["choice"] = int(choice)
        sample["map_elites_ensemble"] = result["selected_method"]
        selected.append(sample)
        counts[int(choice)] += 1
    _atomic_dump(selected, output)

    summary = {key: value for key, value in result.items()
               if key != "choices"}
    summary.update({
        "num_records": len(selected),
        "choice_counts": {"0": counts[0], "1": counts[1]},
        "pair_data": str(pair_data),
        "pair_data_sha256": _sha256(pair_data),
        "margin_manifest": str(margin_manifest),
        "margin_manifest_sha256": _sha256(margin_manifest),
        "output": str(output),
        "output_sha256": _sha256(output),
    })
    _atomic_dump(summary, summary_output)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = calibrate_preferences_file(
        args.pair_data, args.margin_manifest, args.output,
        args.summary_output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
