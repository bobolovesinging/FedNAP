"""Generate TLDR candidate pairs with the existing RLHF generator."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-config', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--data-root')
    parser.add_argument('--evaluation', type=Path,
                        help='Require candidate prompts to be disjoint from this test file')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    import torch
    from federatedscope.core.configs.config import global_cfg
    from federatedscope.core.auxiliaries.utils import setup_seed
    from federatedscope.llm.model.model_builder import get_llm
    from federatedscope.llm.dataloader.dataloader import get_tokenizer
    from federatedscope.llm.rlhf.standalone_training import RLHF_finetuning
    cfg = global_cfg.clone()
    cfg.merge_from_file(args.policy_config)
    cfg.merge_from_list(['data.type', 'reddit-tldr-rlhf@llm',
                         'llm.accelerator.use', 'False'])
    if args.data_root:
        cfg.merge_from_list(['data.root', args.data_root])
    cfg.freeze(save=False)
    if args.evaluation:
        from federatedscope.llm.dataloader.reddit_tldr import load_human_finetuning_dataset
        from scripts.prepare_server_validation import prompt_id, prompt_ids

        data_dir = Path(cfg.data.root) / 'reddit-tldr-comparison'
        required = ('candidates.jsonl', 'sft.jsonl', 'policy_val.jsonl', 'policy_dev_holdout.jsonl')
        if not all((data_dir / name).is_file() for name in required):
            raise FileNotFoundError('Run stage_tldr.py before generating candidates')
        evaluation_prompts = prompt_ids(args.evaluation, 'post')
        # Check the exact train slice that the generator will read.
        prompts, _, _ = load_human_finetuning_dataset(
            str(data_dir), None, rlhf=True, raw_no_prompt=True)
        candidate_keys = {prompt_id(row, 'post') for row in prompts}
        if candidate_keys & evaluation_prompts:
            raise ValueError('TLDR candidate source overlaps the evaluation file')
    setup_seed(cfg.seed)
    model_name = cfg.model.type.split('@')[0]
    tokenizer, _ = get_tokenizer(model_name, cfg.data.root, cfg.llm.tok_len, padding_side='left')
    model = get_llm(cfg)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])
    model.to('cuda:0').eval()
    trainer = RLHF_finetuning(model=model, tokenizer=tokenizer, config=cfg,
                              generator_tokenizer=tokenizer, device='cuda:0')
    records = trainer.load_pairwise_data(use_cache=not args.evaluation)
    if args.evaluation:
        from collections import Counter
        from math import comb
        expected = comb(max(2, cfg.llm.num_completions), 2)
        counts = Counter(prompt_id(row, 'post') for row in records)
        if counts != dict.fromkeys(candidate_keys, expected):
            raise ValueError('Generated pairs do not cover the staged candidate pool exactly')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(records, stream)
    print(json.dumps({'output': str(output), 'pairs': len(records)}))


if __name__ == '__main__':
    main()
