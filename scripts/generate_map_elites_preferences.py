#!/usr/bin/env python
"""Generate TLDR preference pairs with four MAP-Elites selectors."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--pair-data", required=True)
    parser.add_argument("--cell-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", choices=("tldr", "shp"), default="tldr")
    parser.add_argument("--work-dir")
    parser.add_argument(
        "--tie-break-mode",
        choices=("client_count", "reliability_weighted"),
        default="client_count",
    )
    return parser.parse_args()


def selector_prompt_for_task(task):
    if task == 'tldr':
        from federatedscope.llm.dataloader.reddit_tldr import TLDR_PROMPT_DICT
        return TLDR_PROMPT_DICT['summary_cmp']
    if task == 'shp':
        from federatedscope.llm.dataset.shp_preference import \
            SHP_COMPARISON_PROMPT
        return SHP_COMPARISON_PROMPT
    raise ValueError(f'unsupported selector task: {task}')


def load_json(path):
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def atomic_dump_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_token_ids(encoded):
    ids = encoded["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(value) for value in ids]


def _encode(tokenizer, text, *, add_special_tokens):
    return _flat_token_ids(tokenizer(
        text,
        add_special_tokens=add_special_tokens,
        truncation=False,
    ))


def _balanced_allocations(lengths, budget):
    allocations = [0 for _ in lengths]
    remaining = max(int(budget), 0)
    while remaining:
        progressed = False
        for index in sorted(range(len(lengths)),
                            key=lambda value: (allocations[value], value)):
            if allocations[index] >= lengths[index]:
                continue
            allocations[index] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return allocations


def _truncate_shp_source(record, tokenizer, selector_prompt,
                         max_source_tokens):
    fields = ("instruction", "output_A", "output_B")
    source = selector_prompt.format_map(record)
    source_ids = _encode(tokenizer, source, add_special_tokens=True)
    if len(source_ids) <= max_source_tokens:
        return source_ids, False

    field_ids = [
        _encode(tokenizer, str(record[field]), add_special_tokens=False)
        for field in fields
    ]
    empty = dict(record)
    for field in fields:
        empty[field] = ""
    fixed_size = len(_encode(
        tokenizer,
        selector_prompt.format_map(empty),
        add_special_tokens=True,
    ))
    # Leave a small boundary-token allowance because independently decoded
    # fields can tokenize differently when inserted back into the template.
    allocations = _balanced_allocations(
        [len(value) for value in field_ids],
        max_source_tokens - fixed_size - 8,
    )

    for _ in range(8):
        truncated = dict(record)
        for field, values, count in zip(fields, field_ids, allocations):
            truncated[field] = tokenizer.decode(
                values[:count], skip_special_tokens=True)
        source_ids = _encode(
            tokenizer,
            selector_prompt.format_map(truncated),
            add_special_tokens=True,
        )
        excess = len(source_ids) - max_source_tokens
        if excess <= 0:
            return source_ids, True
        for _ in range(excess + 2):
            candidates = [index for index, count in enumerate(allocations)
                          if count > 0]
            if not candidates:
                break
            index = max(candidates, key=lambda value: allocations[value])
            allocations[index] -= 1
    raise ValueError("SHP selector prompt cannot fit the configured context")


class _ShpChoiceInferenceDataset:
    """SHP A/B inputs with one target token retained for every record."""

    def __init__(self, records, tokenizer, selector_prompt):
        import torch

        choice_id = _encode(
            tokenizer, ": A", add_special_tokens=True)[-1]
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("selector tokenizer must define eos_token_id")
        max_length = int(tokenizer.model_max_length)
        if max_length <= 2:
            raise ValueError("selector context is too short")

        self.input_ids = []
        self.labels = []
        self.categories = []
        self.truncated_count = 0
        for record in records:
            source_ids, was_truncated = _truncate_shp_source(
                record, tokenizer, selector_prompt, max_length - 2)
            input_ids = source_ids + [choice_id, int(eos_id)]
            labels = [-100] * len(input_ids)
            labels[-2] = choice_id
            self.input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            self.labels.append(torch.tensor(labels, dtype=torch.long))
            self.categories.append(0)
            self.truncated_count += int(was_truncated)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "labels": self.labels[index],
            "categories": self.categories[index],
        }


def build_token_dataset(records, tokenizer, selector_prompt, task="tldr"):
    from federatedscope.llm.dataset.llm_dataset import LLMDataset

    if task == "shp":
        return _ShpChoiceInferenceDataset(
            records, tokenizer, selector_prompt)
    if task != "tldr":
        raise ValueError(f"unsupported selector task: {task}")
    for sample in records:
        sample["fake_choice"] = " A"
    try:
        return LLMDataset(
            records,
            tokenizer,
            prompt_input=selector_prompt,
            prompt_no_input=selector_prompt,
            output_tag="fake_choice",
        )
    finally:
        for sample in records:
            sample.pop("fake_choice", None)


def choices_and_margins_from_logits(choice_logits):
    """Convert A/B logits to hard choices and signed B-minus-A margins."""
    if choice_logits.ndim != 2 or choice_logits.shape[1] != 2:
        raise ValueError("choice logits must have shape [records, 2]")
    if not choice_logits.isfinite().all():
        raise ValueError("choice logits contain non-finite values")
    predictions = choice_logits.argmax(dim=1)
    margins = choice_logits[:, 1] - choice_logits[:, 0]
    return (
        [int(value) for value in predictions.tolist()],
        [float(value) for value in margins.float().tolist()],
    )


def predict_choices_and_margins(model, token_dataset, tokenizer, batch_size,
                                device):
    import torch
    from torch.utils.data import DataLoader

    from federatedscope.llm.dataloader.dataloader import LLMDataCollator
    from federatedscope.llm.rlhf.standalone_training import cal_acc

    choice_token_ids = [
        tokenizer(f": {choice}")["input_ids"][-1]
        for choice in ("A", "B")
    ]
    dataloader = DataLoader(
        token_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=LLMDataCollator(tokenizer=tokenizer),
    )

    predictions = []
    margins = []
    model.eval()
    with torch.inference_mode():
        for batch_index, data_batch in enumerate(dataloader):
            input_ids = data_batch["input_ids"].to(device)
            labels = data_batch["labels"].to(device)
            attention_mask = data_batch["attention_mask"].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            _, choice_logits, _, _ = cal_acc(
                outputs.logits, labels, choice_token_ids)
            batch_predictions, batch_margins = \
                choices_and_margins_from_logits(choice_logits)
            if len(batch_predictions) != input_ids.shape[0]:
                raise ValueError(
                    "selector batch did not produce exactly one choice "
                    "logit pair per record")
            predictions.extend(batch_predictions)
            margins.extend(batch_margins)
            if (batch_index + 1) % 100 == 0:
                print(
                    f"inference batches={batch_index + 1} "
                    f"records={len(predictions)}",
                    flush=True,
                )
    return predictions, margins


def predict_choices(model, token_dataset, tokenizer, batch_size, device):
    predictions, _ = predict_choices_and_margins(
        model, token_dataset, tokenizer, batch_size, device)
    return predictions


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_records is not None and args.max_records <= 0:
        raise ValueError("--max-records must be positive")

    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.llm.dataloader.dataloader import get_tokenizer
    from federatedscope.llm.map_elites_voting import (
        build_voted_preferences,
        cells_from_manifest,
        load_cell_checkpoint,
    )
    from federatedscope.llm.model.model_builder import get_llm

    cell_dir = Path(args.cell_dir)
    manifest_path = cell_dir / "manifest.json"
    manifest = load_json(manifest_path)
    cells = cells_from_manifest(manifest)

    records = load_json(args.pair_data)
    if args.max_records is not None:
        records = records[:args.max_records]
    required_fields = {"output_A", "output_B"}
    for index, sample in enumerate(records):
        missing = required_fields.difference(sample)
        if missing:
            raise ValueError(
                f"pair record {index} is missing fields: {sorted(missing)}")

    selector_cfg = global_cfg.clone()
    selector_cfg.merge_from_file(args.selector_config)
    selector_cfg.freeze(save=False)
    setup_seed(selector_cfg.seed)

    model_name, _ = selector_cfg.model.type.split("@")
    model = get_llm(selector_cfg, device_map="auto")
    tokenizer, _ = get_tokenizer(
        model_name,
        selector_cfg.data.root,
        selector_cfg.llm.tok_len,
    )
    token_dataset = build_token_dataset(
        records, tokenizer, selector_prompt_for_task(args.task),
        task=args.task)

    output_path = Path(args.output)
    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else output_path.with_suffix(output_path.suffix + ".work")
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    input_key = (sha256_file(Path(args.pair_data)), sha256_file(Path(args.selector_config)),
                 args.task, len(records), sha256_file(manifest_path))
    votes_by_cell = []
    for cell in cells:
        cell_id = cell["cell_id"]
        cell_name = f"{cell_id[0]}_{cell_id[1]}"
        checkpoint_path = cell_dir / cell["checkpoint"]
        load_cell_checkpoint(model, checkpoint_path, expected_cell=cell)
        cache_key = hashlib.sha256(repr((*input_key, sha256_file(checkpoint_path))).encode()).hexdigest()
        vote_cache_path = work_dir / f"votes_{cell_name}_{cache_key}.json"

        if vote_cache_path.exists():
            votes = load_json(vote_cache_path)
            if len(votes) != len(records):
                raise ValueError(
                    f"incomplete vote cache: {vote_cache_path}")
            print(f"reuse vote cache {vote_cache_path}", flush=True)
        else:
            print(f"load and infer cell {cell_id}", flush=True)
            votes = predict_choices(
                model,
                token_dataset,
                tokenizer,
                args.batch_size,
                args.device,
            )
            if len(votes) != len(records):
                raise ValueError(
                    f"cell {cell_id} produced {len(votes)} votes for "
                    f"{len(records)} records")
            atomic_dump_json(votes, vote_cache_path)
        votes_by_cell.append(votes)

    selected, summary = build_voted_preferences(
        records,
        votes_by_cell,
        cells,
        tie_break_mode=args.tie_break_mode,
    )
    atomic_dump_json(selected, output_path)
    summary.update({
        "manifest": str(manifest_path),
        "pair_data": str(Path(args.pair_data)),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "cells": cells,
    })
    atomic_dump_json(summary, args.summary_output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
