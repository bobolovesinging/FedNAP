#!/usr/bin/env python3
"""Materialize the upstream SHP comparison rows for scripts.stage_shp."""

import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path, records):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        for record in records:
            stream.write(json.dumps(
                record, ensure_ascii=True, sort_keys=True) + '\n')
    temporary.replace(path)


def prepare(output_dir, client_num=200, alpha=0.3, seed=0,
            dataset_revision=None, tokenizer_path=None,
            max_pair_tokens=512):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from federatedscope.llm.dataset.shp_preference import (
        assign_shp_prompts_to_clients,
        filter_shp_records_by_token_length,
        prepare_shp_split,
        FIRST_PERSON_VERSION,
    )

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = dataset_revision or HfApi().dataset_info(
        'stanfordnlp/SHP').sha
    dataset = load_dataset('stanfordnlp/SHP', revision=resolved_revision)
    if not tokenizer_path:
        raise ValueError('tokenizer_path is required for baseline parity')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True)
    train_rows = list(dataset['train'])
    assignments = assign_shp_prompts_to_clients(
        ({'instruction': row['history'], 'domain': row['domain']}
         for row in train_rows),
        client_num=client_num,
        alpha=alpha,
        seed=seed,
    )
    split_aliases = {
        'train': 'train',
        'validation': 'validation',
        'test': 'test',
    }
    files = {}
    counts = {}
    for source_split, output_split in split_aliases.items():
        records = prepare_shp_split(
            train_rows if source_split == 'train'
            else dataset[source_split],
            split=output_split,
            client_assignments=(assignments
                                if source_split == 'train' else None),
        )
        if source_split == 'train':
            records = filter_shp_records_by_token_length(
                records, tokenizer, max_pair_tokens)
        path = output_dir / f'shp_{output_split}.jsonl'
        _write_jsonl(path, records)
        files[path.name] = _sha256(path)
        counts[output_split] = len(records)

    assignment_path = output_dir / 'client_assignments.json'
    assignment_path.write_text(json.dumps(
        assignments, indent=2, sort_keys=True), encoding='utf-8')
    files[assignment_path.name] = _sha256(assignment_path)
    client_counts = Counter(assignments.values())
    # Recompute filtered client counts from the materialized file to validate
    # the exact population consumed by all three selectors.
    with (output_dir / 'shp_train.jsonl').open(encoding='utf-8') as stream:
        filtered_client_counts = Counter(
            int(json.loads(line)['category'].split('_')[-1])
            for line in stream)
    if len(filtered_client_counts) != int(client_num) \
            or min(filtered_client_counts.values()) <= 0:
        raise RuntimeError('token filtering produced an empty SHP client')
    manifest = {
        'complete': True,
        'dataset': 'stanfordnlp/SHP',
        'dataset_revision': resolved_revision,
        'label_semantics': 'SHP labels=1 means A wins; internal choice=0 means A wins',
        'first_person_version': FIRST_PERSON_VERSION,
        'first_person_definition': 'chosen rate > rejected rate; exact pronoun/contraction '
                                   'tokens; normalize curly apostrophes; ties remain zero',
        'client_num': int(client_num),
        'dirichlet_alpha': float(alpha),
        'seed': int(seed),
        'tokenizer_path': str(Path(tokenizer_path).resolve()),
        'max_pair_tokens': int(max_pair_tokens),
        'counts': counts,
        'client_prompt_counts': {
            str(index): client_counts[index] for index in range(client_num)
        },
        'train_client_record_counts': {
            str(index): filtered_client_counts[index]
            for index in range(client_num)
        },
        'files': files,
    }
    manifest_path = output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(
        manifest, indent=2, sort_keys=True), encoding='utf-8')
    (output_dir / 'PREPARED_COMPLETE').write_text(
        resolved_revision + '\n', encoding='ascii')
    return manifest


if __name__ == '__main__':
    raise SystemExit('Use python -m scripts.stage_shp to build all SHP roles together.')
