#!/usr/bin/env python
"""Cache validation and unlabeled A/B margins for four MAP-Elites cells."""

import argparse
import hashlib
import json
from pathlib import Path

from scripts.generate_map_elites_preferences import (
    atomic_dump_json,
    build_token_dataset,
    load_json,
    predict_choices_and_margins,
    selector_prompt_for_task,
    sha256_file,
)


def _panel_key(item, task="tldr"):
    if task == "tldr":
        values = [
            item.get("category"), item.get("post"), item.get("title"),
            item.get("output_A"), item.get("output_B"), item.get("choice"),
        ]
    elif task == "shp":
        values = [
            item.get("category"), item.get("instruction"),
            item.get("output_A"), item.get("output_B"), item.get("choice"),
        ]
    else:
        raise ValueError(f"unsupported task: {task}")
    identity = json.dumps(values, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).digest()


def load_public_validation_records(path, size=256, task="tldr"):
    records = []
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    if len(records) < size:
        raise ValueError(
            f"validation source has {len(records)} records, expected {size}")
    return sorted(records, key=lambda item: _panel_key(item, task))[:size]


def cache_margin_artifacts(selector_config, pair_data, validation_data,
                           cell_dir, output_dir, batch_size=10,
                           max_records=None, validation_size=256,
                           task="tldr"):
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.llm.dataloader.dataloader import get_tokenizer
    from federatedscope.llm.map_elites_voting import (
        cells_from_manifest,
        load_cell_checkpoint,
    )
    from federatedscope.llm.model.model_builder import get_llm

    pair_data = Path(pair_data)
    validation_data = Path(validation_data)
    cell_dir = Path(cell_dir)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"margin output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = cell_dir / "manifest.json"
    manifest = load_json(manifest_path)
    cells = cells_from_manifest(manifest)
    pair_records = load_json(pair_data)
    if max_records is not None:
        pair_records = pair_records[:max_records]
        validation_size = min(validation_size, max_records)
    validation_records = load_public_validation_records(
        validation_data, size=validation_size, task=task)
    validation_labels = [int(record["choice"])
                         for record in validation_records]

    selector_cfg = global_cfg.clone()
    selector_cfg.merge_from_file(str(selector_config))
    selector_cfg.freeze(save=False)
    setup_seed(selector_cfg.seed)
    model_name, _ = selector_cfg.model.type.split("@")
    model = get_llm(selector_cfg, device_map="auto")
    tokenizer, _ = get_tokenizer(
        model_name, selector_cfg.data.root, selector_cfg.llm.tok_len)
    prompt = selector_prompt_for_task(task)
    pair_dataset = build_token_dataset(
        pair_records, tokenizer, prompt, task=task)
    validation_dataset = build_token_dataset(
        validation_records, tokenizer, prompt, task=task)
    device = "cuda:0" if selector_cfg.use_gpu else "cpu"

    cache_records = []
    for cell in cells:
        row, column = cell["cell_id"]
        name = f"{row}_{column}"
        checkpoint_path = cell_dir / cell["checkpoint"]
        load_cell_checkpoint(model, checkpoint_path, expected_cell=cell)
        validation_choices, validation_margins = \
            predict_choices_and_margins(
                model, validation_dataset, tokenizer, batch_size,
                device)
        pair_choices, pair_margins = predict_choices_and_margins(
            model, pair_dataset, tokenizer, batch_size, device)

        paths = {
            "validation_choices": output_dir / f"validation_choices_{name}.json",
            "validation_margins": output_dir / f"validation_margins_{name}.json",
            "pair_choices": output_dir / f"pair_choices_{name}.json",
            "pair_margins": output_dir / f"pair_margins_{name}.json",
        }
        values = {
            "validation_choices": validation_choices,
            "validation_margins": validation_margins,
            "pair_choices": pair_choices,
            "pair_margins": pair_margins,
        }
        for key, path in paths.items():
            atomic_dump_json(values[key], path)
        cache_records.append({
            "cell_id": list(cell["cell_id"]),
            "client_count": int(cell["client_count"]),
            "checkpoint": str(checkpoint_path),
            "elite_round": cell["elite_round"],
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "paths": {key: str(path) for key, path in paths.items()},
            "sha256": {key: sha256_file(path) for key, path in paths.items()},
        })

    labels_path = output_dir / "validation_labels.json"
    atomic_dump_json(validation_labels, labels_path)
    cache_manifest = {
        "complete": True,
        "pair_count": len(pair_records),
        "validation_count": len(validation_records),
        "pair_data": str(pair_data),
        "pair_data_sha256": sha256_file(pair_data),
        "validation_data": str(validation_data),
        "validation_data_sha256": sha256_file(validation_data),
        "validation_labels": str(labels_path),
        "validation_labels_sha256": sha256_file(labels_path),
        "selector_config": str(selector_config),
        "selector_config_sha256": sha256_file(selector_config),
        "task": task,
        "cell_manifest": str(manifest_path),
        "cell_manifest_sha256": sha256_file(manifest_path),
        "cells": cache_records,
    }
    manifest_output = output_dir / "margin_manifest.json"
    atomic_dump_json(cache_manifest, manifest_output)
    return cache_manifest


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--validation-data", required=True)
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--validation-size", type=int, default=256)
    parser.add_argument("--task", choices=("tldr", "shp"), default="tldr")
    return parser.parse_args()


def main():
    args = parse_args()
    result = cache_margin_artifacts(
        args.selector_config, args.pair_data, args.validation_data,
        args.cell_dir, args.output_dir, args.batch_size, args.max_records,
        args.validation_size, task=args.task)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
