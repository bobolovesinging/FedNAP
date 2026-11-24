#!/usr/bin/env python
"""Select SHP candidate pairs with one FedBiscuit selector."""

import argparse
import copy
import json
from collections import Counter

from scripts.generate_map_elites_preferences import (
    atomic_dump_json,
    build_token_dataset,
    load_json,
    predict_choices,
    selector_prompt_for_task,
    sha256_file,
)


def build_preferences(records, choices):
    if len(records) != len(choices):
        raise ValueError("selector choices must cover every candidate pair")
    selected = []
    for record, choice in zip(records, choices):
        if int(choice) not in (0, 1):
            raise ValueError("selector choices must be binary")
        sample = copy.deepcopy(record)
        sample["choice"] = int(choice)
        selected.append(sample)
    return selected


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--selector-checkpoint", required=True)
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("tldr", "shp"), default="shp")
    return parser.parse_args()


def main():
    args = parse_args()
    import torch

    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.llm.dataloader.dataloader import get_tokenizer
    from federatedscope.llm.model.model_builder import get_llm

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    records = load_json(args.pair_data)
    if not records:
        raise ValueError("candidate pair data must not be empty")
    cfg = global_cfg.clone()
    cfg.merge_from_file(args.selector_config)
    cfg.freeze(save=False)
    setup_seed(cfg.seed)
    model_name, _ = cfg.model.type.split("@")
    model = get_llm(cfg, device_map="auto")
    checkpoint = torch.load(
        args.selector_checkpoint, map_location="cpu", weights_only=False)
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict) or not model_state:
        raise ValueError("selector checkpoint has no model state")
    if not all(torch.isfinite(value).all().item()
               for value in model_state.values() if torch.is_tensor(value)):
        raise FloatingPointError("selector checkpoint contains non-finite values")
    model.load_state_dict(model_state, strict=False)
    tokenizer, _ = get_tokenizer(
        model_name, cfg.data.root, cfg.llm.tok_len)
    dataset = build_token_dataset(
        records, tokenizer, selector_prompt_for_task(args.task),
        task=args.task)
    choices = predict_choices(
        model, dataset, tokenizer, args.batch_size, args.device)
    selected = build_preferences(records, choices)
    atomic_dump_json(selected, args.output)
    counts = Counter(row["choice"] for row in selected)
    summary = {
        "complete": True,
        "record_count": len(selected),
        "choice_counts": {"0": counts[0], "1": counts[1]},
        "selector_checkpoint": args.selector_checkpoint,
        "selector_checkpoint_sha256": sha256_file(args.selector_checkpoint),
        "pair_data": args.pair_data,
        "pair_data_sha256": sha256_file(args.pair_data),
        "output": args.output,
        "output_sha256": sha256_file(args.output),
    }
    atomic_dump_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
