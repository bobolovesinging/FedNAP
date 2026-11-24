import argparse
import json
import os

import torch
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.configs.config import global_cfg
from federatedscope.llm.dataloader.dataloader import get_tokenizer, load_jsonl
from federatedscope.llm.dataloader.reddit_tldr import TLDR_PROMPT_DICT
from federatedscope.llm.model.model_builder import get_llm

torch.compile = lambda model=None, *a, **k: (lambda m: m) \
    if model is None else model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--output-prefix", default="final_")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--model-cache", default="")
    parser.add_argument("--test-file")
    parser.add_argument("--expected-test-sha256")
    args = parser.parse_args()

    cfg = global_cfg.clone()
    cfg.merge_from_file(args.cfg)
    test_file = args.test_file or os.path.join(cfg.data.root, "reddit-tldr_test.jsonl")
    if args.expected_test_sha256:
        from scripts.prepare_server_validation import file_sha256
        from pathlib import Path
        if file_sha256(Path(test_file)) != args.expected_test_sha256.lower():
            parser.error("TLDR test file does not match the expected SHA-256")
    cfg.merge_from_list([
        "llm.accelerator.use", "False",
        "federate.save_to", args.ckpt,
        "train.is_enable_half", "True",
    ])
    if args.model_cache:
        cfg.merge_from_list(["llm.cache.model", args.model_cache])
    update_logger(cfg, clear_before_add=True)
    setup_seed(cfg.seed)
    cfg.freeze()

    os.makedirs(args.outdir, exist_ok=True)
    model_name, _ = cfg.model.type.split("@")
    tokenizer, _ = get_tokenizer(model_name, cfg.data.root, cfg.llm.tok_len,
                                 padding_side="left")

    print("load model", flush=True)
    model = get_llm(cfg)
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    print("loaded round", ckpt.get("cur_round"), flush=True)
    model.to("cuda:0")
    model.eval()

    data = load_jsonl(test_file,
                      subreddit="subreddit",
                      title="title",
                      post="post",
                      summary="summary")
    prompt = TLDR_PROMPT_DICT["summary"]
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    out_txt = os.path.join(args.outdir,
                           f"{args.output_prefix}_summarization.txt")
    out_json = os.path.join(args.outdir, "summarization.json")
    empty_count = 0
    print("num samples", len(data), "out", out_txt, flush=True)

    with open(out_txt, "w", encoding="utf-8") as f:
        for left in tqdm(range(0, len(data), args.batch_size)):
            batch = data[left:left + args.batch_size]
            input_texts = [prompt.format_map(x) for x in batch]
            inputs = tokenizer(input_texts,
                               padding=True,
                               add_special_tokens=True,
                               return_tensors="pt").to("cuda:0")
            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    top_p=1.0,
                    temperature=0.0,
                    do_sample=False,
                    max_new_tokens=cfg.llm.max_new_token)
            completions = tokenizer.batch_decode(
                output_ids[:, input_len:],
                skip_special_tokens=True,
                ignore_tokenization_space=True)
            completions = [x.strip().replace("</s>", "") for x in completions]

            for sample, comp in zip(batch, completions):
                if not comp.strip():
                    empty_count += 1
                sample["completion"] = comp
                score = scorer.score(sample["summary"], comp)
                sample["score"] = score
                aggregator.add_scores(score)
                f.write(f'Subreddit: r/{sample["subreddit"]}\n\n')
                f.write(f'Title:\n{sample["title"]}\n\n')
                f.write(f'Post:\n{sample["post"]}\n\n')
                f.write(f'Human summary:\n{sample["summary"]}\n\n')
                f.write(f'Model-generated summary 0:\n{comp}\n\n')
                f.write(f'Score:\n{score}\n\n')
                f.write("==========================\n\n")
            f.flush()
        f.write(json.dumps(aggregator.aggregate()) + "\n")

    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(data, jf)

    print("empty completions", empty_count, flush=True)
    print("done", out_txt, flush=True)


if __name__ == "__main__":
    main()
