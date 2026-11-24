"""Evaluate the public SFT+DPO adapters on the pinned TLDR test set."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from prepare_server_validation import file_sha256
from tldr_benchmark import TEST_SHA

PINNED = {
    "checkpoints/tldr_sft.ckpt": "62676df9357da20b41cabc3000cf681af0ed07d5842ea3f64a6f2ea2310dd23d",
    "checkpoints/tldr_dpo.ckpt": "0084a869c5951ad990c2310e5983a266c2af82c3436e4db840e4640f390a0932",
    "evidence/base-model.sha256": "46b03f425205b93e5de42e52855c6138d0b9ef75663113860afd01dd1b18b05f",
    "evidence/original_sft.yaml": "ba67854871f204efa30ace81e26504e1ec72bfa23f6a186c2acf610f61e58011",
    "evidence/original_dpo.yaml": "3d42ac809cc788f6b37c5001f794359bc721dcfcf51255e0e87a45279e2617dc"
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', default='release-assets')
    parser.add_argument('--base-model')
    parser.add_argument('--test-file', type=Path, required=True)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument('--verify-inputs', action='store_true',
                        help='Verify the public adapters, configs and official test on CPU')
    parser.add_argument('--output-dir', default='outputs/public-tldr-eval')
    parser.add_argument('--batch-size', type=int, default=20)
    checks.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    root = Path(args.artifacts).resolve()
    test_file = args.test_file.resolve()
    if file_sha256(test_file) != TEST_SHA:
        raise ValueError('Official test file checksum mismatch')
    for name, sha in PINNED.items():
        if file_sha256(root / name) != sha:
            raise ValueError(f'Public artifact checksum mismatch: {name}')
    if args.verify_inputs:
        print('Verified both adapters, configs, base-model manifest and fixed test.')
        return
    if not args.base_model:
        parser.error('--base-model is required for generation or --check-only')
    base = Path(args.base_model).resolve()
    # Pin the actual ModelScope mirror files used by the saved run.
    for line in (root / 'evidence/base-model.sha256').read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        if file_sha256(base / name.strip()) != sha:
            raise ValueError(f'Base-model hash mismatch: {name.strip()}')
    if args.check_only:
        print('Checkpoint, test-set and base-model hashes match the public evaluation.')
        return
    import yaml
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sft = yaml.safe_load((root / 'evidence/original_sft.yaml').read_text())
    dpo = yaml.safe_load((root / 'evidence/original_dpo.yaml').read_text())
    model_type = str(base) + '@huggingface_llm'
    for config in (sft, dpo):
        config['model']['type'] = model_type
        config['data']['root'] = str(test_file.parent)
        config['outdir'] = str(output / 'runtime')
    sft_path = output / 'sft.yaml'
    dpo_path = output / 'dpo.yaml'
    dpo['model']['load_from_local_pretrained_fs_config'] = str(sft_path)
    dpo['model']['load_from_local_pretrained_model_path'] = str(root / 'checkpoints/tldr_sft.ckpt')
    sft_path.write_text(yaml.safe_dump(sft), encoding='utf-8')
    dpo_path.write_text(yaml.safe_dump(dpo), encoding='utf-8')
    command = [sys.executable, 'scripts/eval_tldr_policy_direct_leftpad.py',
               '--cfg', str(dpo_path), '--ckpt', str(root / 'checkpoints/tldr_dpo.ckpt'),
               '--test-file', str(test_file), '--expected-test-sha256', TEST_SHA,
               '--outdir', str(output), '--batch-size', str(args.batch_size)]
    environment = dict(os.environ, FEDBISCUIT_FORCE_BF16='1',
                       PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
    subprocess.run(command, check=True, env=environment,
                   cwd=Path(__file__).parents[1])


if __name__ == '__main__':
    main()
