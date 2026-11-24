#!/usr/bin/env python
"""Build gamma-weighted SHP preferences from calibrated cell margins."""

import argparse
import copy
import hashlib
import json
import math
from collections import Counter

from federatedscope.llm.map_elites_improvement import (
    confidence_from_logit,
    select_calibrated_ensemble,
)
from scripts.generate_map_elites_preferences import atomic_dump_json


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def verified_load(path, expected_sha256):
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact hash mismatch: {path}")
    return load_json(path)


def apply_confidence_weights(records, logits, gamma=0.5):
    if len(records) != len(logits) or not records:
        raise ValueError("records and calibrated logits must be aligned")
    gamma = float(gamma)
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("confidence gamma must be finite and positive")
    selected = []
    for record, logit in zip(records, logits):
        value = float(logit)
        if not math.isfinite(value):
            raise ValueError("calibrated logits must be finite")
        confidence = confidence_from_logit(value)
        sample = copy.deepcopy(record)
        sample.update({
            "choice": 1 if value >= 0.0 else 0,
            "calibrated_logit": value,
            "confidence": confidence,
            "confidence_gamma": gamma,
            "dpo_weight": max(0.25, confidence ** gamma),
        })
        selected.append(sample)
    return selected


def build_from_manifest(pair_data, margin_manifest, output, summary_output,
                        gamma=0.5):
    cache = load_json(margin_manifest)
    if not cache.get("complete") or len(cache.get("cells", [])) != 4:
        raise ValueError("margin manifest is incomplete")
    if sha256_file(pair_data) != cache["pair_data_sha256"]:
        raise ValueError("pair data hash does not match margin cache")
    records = load_json(pair_data)[:int(cache["pair_count"])]
    labels = verified_load(
        cache["validation_labels"], cache["validation_labels_sha256"])
    validation_margins = []
    pair_margins = []
    client_counts = []
    for cell in cache["cells"]:
        validation_margins.append(verified_load(
            cell["paths"]["validation_margins"],
            cell["sha256"]["validation_margins"]))
        pair_margins.append(verified_load(
            cell["paths"]["pair_margins"],
            cell["sha256"]["pair_margins"]))
        client_counts.append(int(cell["client_count"]))
    result = select_calibrated_ensemble(
        validation_margins, labels, pair_margins, client_counts)
    selected = apply_confidence_weights(
        records, result["pair_logits"], gamma=gamma)
    atomic_dump_json(selected, output)
    counts = Counter(row["choice"] for row in selected)
    weights = [row["dpo_weight"] for row in selected]
    summary = {
        "complete": True,
        "record_count": len(selected),
        "choice_counts": {"0": counts[0], "1": counts[1]},
        "gamma": float(gamma),
        "weight_min": min(weights),
        "weight_max": max(weights),
        "weight_mean": sum(weights) / len(weights),
        "selected_method": result["selected_method"],
        "selected_l2": result["selected_l2"],
        "temperatures": result["temperatures"],
        "parameters": result["parameters"],
        "pair_data": pair_data,
        "pair_data_sha256": sha256_file(pair_data),
        "margin_manifest": margin_manifest,
        "margin_manifest_sha256": sha256_file(margin_manifest),
        "output": output,
        "output_sha256": sha256_file(output),
    }
    atomic_dump_json(summary, summary_output)
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--margin-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--gamma", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_from_manifest(
        args.pair_data, args.margin_manifest, args.output,
        args.summary_output, gamma=args.gamma)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
