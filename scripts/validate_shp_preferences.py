#!/usr/bin/env python
"""Validate SHP preference artifacts before formal DPO starts."""

import argparse
import hashlib
import json
import math


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_preferences(path, *, expected_policy, expected_source,
                         require_weights=False, expected_gamma=None):
    with open(path, "r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("preference file must be a non-empty JSON list")
    required = {
        "instruction", "output_A", "output_B", "choice", "source",
        "source_sha256", "policy", "policy_revision",
    }
    source_hashes = set()
    revisions = set()
    for index, record in enumerate(records):
        missing = required.difference(record)
        if missing:
            raise ValueError(
                f"preference record {index} is missing {sorted(missing)}")
        if record["policy"] != expected_policy:
            raise ValueError("preference policy does not match expected policy")
        if record["source"] != expected_source:
            raise ValueError("preference source does not match expected source")
        if record["choice"] not in (0, 1):
            raise ValueError("preference choice must be binary")
        source_hashes.add(record["source_sha256"])
        revisions.add(record["policy_revision"])
        has_weight = "dpo_weight" in record
        if has_weight != require_weights:
            raise ValueError("dpo_weight coverage does not match method")
        if require_weights:
            weight = float(record["dpo_weight"])
            if not math.isfinite(weight):
                raise ValueError("dpo_weight must be finite")
            if not 0.0 < weight <= 1.0:
                raise ValueError("dpo_weight must be in (0, 1]")
            gamma = float(record.get("confidence_gamma", float("nan")))
            if not math.isfinite(gamma):
                raise ValueError("confidence gamma must be finite")
            if expected_gamma is not None and gamma != expected_gamma:
                raise ValueError("confidence gamma does not match")
    if len(source_hashes) != 1:
        raise ValueError("mixed source hashes in preference file")
    if len(revisions) != 1:
        raise ValueError("mixed policy revisions in preference file")
    return {
        "record_count": len(records),
        "sha256": sha256_file(path),
        "policy": expected_policy,
        "policy_revision": next(iter(revisions)),
        "source": expected_source,
        "source_sha256": next(iter(source_hashes)),
        "weighted": require_weights,
        "gamma": expected_gamma if require_weights else None,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference-data", required=True)
    parser.add_argument("--policy", choices=("gemma", "llama"), required=True)
    parser.add_argument("--source", choices=(
        "reddit-2", "reddit-4", "saferlhf-2"), required=True)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--expected-gamma", type=float)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = validate_preferences(
        args.preference_data, expected_policy=args.policy,
        expected_source=args.source, require_weights=args.require_weights,
        expected_gamma=args.expected_gamma)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

