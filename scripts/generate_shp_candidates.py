#!/usr/bin/env python
"""Generate policy-specific SHP candidate pairs with immutable provenance."""

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_dump_json(value, path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pair_completions(prompt, completions, *, source, source_sha256,
                     policy, policy_revision):
    """Return all unordered completion pairs for one prompt."""
    if len(completions) < 2:
        raise ValueError("at least two completions are required")
    cleaned = [str(value).strip() for value in completions]
    if any(not value for value in cleaned):
        raise ValueError("candidate completions must be non-empty")
    required = {"prompt_id", "instruction"}
    missing = required.difference(prompt)
    if missing:
        raise ValueError(f"prompt is missing fields: {sorted(missing)}")
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a SHA256 digest")

    records = []
    for index_a, index_b in itertools.combinations(range(len(cleaned)), 2):
        record = copy.deepcopy(prompt)
        record.update({
            "output_A": cleaned[index_a],
            "output_B": cleaned[index_b],
            "completion_index_A": index_a,
            "completion_index_B": index_b,
            "source": source,
            "source_sha256": source_sha256,
            "policy": policy,
            "policy_revision": policy_revision,
        })
        records.append(record)
    return records


def generate_completions(model, tokenizer, prompts, num_completions,
                         batch_size, max_new_tokens, seed):
    import torch

    from federatedscope.llm.dataloader.shp import SHP_PROMPT_DICT

    if num_completions not in (2, 4):
        raise ValueError("SHP Table VII supports two or four completions")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    torch.manual_seed(seed)
    model.eval()
    results = []
    template = SHP_PROMPT_DICT["shp"]
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        texts = [template.format_map(row) for row in batch]
        tokens = tokenizer(
            texts, padding=True, add_special_tokens=True,
            return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            outputs = model.generate(
                **tokens,
                do_sample=True,
                temperature=0.7,
                top_p=1.0,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_completions,
            )
        suffix = outputs[:, tokens["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(suffix, skip_special_tokens=True)
        expected = len(batch) * num_completions
        if len(decoded) != expected:
            raise RuntimeError(
                f"generation returned {len(decoded)} rows, expected {expected}")
        for offset in range(len(batch)):
            lower = offset * num_completions
            upper = lower + num_completions
            values = [value.strip() for value in decoded[lower:upper]]
            if any(not value for value in values):
                raise RuntimeError("generation produced an empty completion")
            results.append(values)
        print(f"generated prompts={len(results)}/{len(prompts)}", flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-data", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--policy", choices=("gemma", "llama"), required=True)
    parser.add_argument("--policy-revision", required=True)
    parser.add_argument("--source", choices=(
        "reddit-2", "reddit-4", "saferlhf-2"), required=True)
    parser.add_argument("--num-completions", type=int, choices=(2, 4),
                        required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def configure_candidate_precision(policy):
    # Match the locked LLaMA DPO precision before the shared loader is called.
    if policy == "llama":
        os.environ["FEDBISCUIT_FORCE_BF16"] = "1"


def main():
    args = parse_args()
    configure_candidate_precision(args.policy)
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.llm.dataloader.dataloader import get_tokenizer
    from federatedscope.llm.model.model_builder import get_llm

    prompt_path = Path(args.prompt_data)
    prompts = json.loads(prompt_path.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompt data must be a non-empty JSON list")
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("max_records must be positive")
        prompts = prompts[:args.max_records]
    expected_completions = 4 if args.source == "reddit-4" else 2
    if args.num_completions != expected_completions:
        raise ValueError("source and num_completions do not match")

    cfg = global_cfg.clone()
    cfg.merge_from_file(args.policy_config)
    cfg.freeze(save=False)
    setup_seed(args.seed)
    model_name, _ = cfg.model.type.split("@")
    model = get_llm(cfg, device_map="auto")
    parameter_dtype = str(next(model.parameters()).dtype)
    if args.policy == "llama" and parameter_dtype != "torch.bfloat16":
        raise RuntimeError(f"LLaMA candidates require BF16, got {parameter_dtype}")
    print(f"CANDIDATE_MODEL_DTYPE={parameter_dtype}", flush=True)
    tokenizer, _ = get_tokenizer(
        model_name, cfg.data.root, cfg.llm.tok_len, padding_side="left")
    generated = generate_completions(
        model, tokenizer, prompts, args.num_completions,
        args.batch_size, args.max_new_tokens, args.seed)

    source_sha256 = sha256_file(prompt_path)
    pairs = []
    for prompt, completions in zip(prompts, generated):
        pairs.extend(pair_completions(
            prompt, completions, source=args.source,
            source_sha256=source_sha256, policy=args.policy,
            policy_revision=args.policy_revision))
    expected_pairs = len(prompts) * (
        args.num_completions * (args.num_completions - 1) // 2)
    if len(pairs) != expected_pairs:
        raise RuntimeError("candidate pair count is incomplete")
    atomic_dump_json(pairs, args.output)
    manifest = {
        "complete": True,
        "model_dtype": parameter_dtype,
        "policy": args.policy,
        "policy_revision": args.policy_revision,
        "source": args.source,
        "source_prompt_file": str(prompt_path),
        "source_sha256": source_sha256,
        "prompt_count": len(prompts),
        "num_completions": args.num_completions,
        "pair_count": len(pairs),
        "seed": args.seed,
        "temperature": 0.7,
        "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
        "output": str(Path(args.output)),
        "output_sha256": sha256_file(args.output),
    }
    atomic_dump_json(manifest, args.manifest_output)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
