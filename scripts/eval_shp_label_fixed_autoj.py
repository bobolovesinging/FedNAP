"""Two-position Auto-J evaluation against the pinned SHP reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path

REFERENCE_SHA256 = '83db546b872ddebee8965fd05fa48461ee3c32bc695c62fb57f2d214ff741ec4'


def unwrap(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError('missing model or reference response')
    return value


def outcome(label, position):
    if label == 2:
        return 'tie'
    if label not in (0, 1):
        return 'invalid'
    return 'win' if label == position - 1 else 'loss'


def summarize(rows):
    counts = {key: sum(row['outcome'] == key for row in rows)
              for key in ('win', 'loss', 'tie', 'invalid')}
    valid = len(rows) - counts['invalid']
    if valid / len(rows) < 0.9:
        raise RuntimeError('Auto-J valid coverage below 90%')
    return {**counts, 'count': len(rows), 'valid_count': valid,
            'strict_win_rate': 100 * counts['win'] / valid,
            'half_tie_win_rate': 100 * (counts['win'] + 0.5 * counts['tie']) / valid}


def run(root, policy, method, reference_path, generation=None, output_dir=None):
    from vllm import SamplingParams
    from federatedscope.llm.eval.eval_for_tldr.auto_j_vllm import (
        build_llm, extract_pariwise_result)
    from federatedscope.llm.eval.eval_for_tldr.auto_j_constants_prompt import build_autoj_input

    if generation is None:
        run_root = root / 'results/downstream' / policy / 'reddit-2' / method
        if not (run_root / 'DOWNSTREAM_COMPLETE').is_file():
            raise FileNotFoundError('Legacy downstream completion marker is missing')
        generation = run_root / 'alpaca_eval/alpaca_eval.json'
    outputs = json.loads(generation.read_text(encoding='utf-8'))
    reference_bytes = reference_path.read_bytes()
    reference_sha = hashlib.sha256(reference_bytes).hexdigest()
    if reference_sha != REFERENCE_SHA256:
        raise ValueError('SHP reference SHA-256 differs from the archived evaluation')
    references = json.loads(reference_bytes)
    ref = {item['instruction']: unwrap(item['output']) for item in references}
    assert len(outputs) == len(references) == len(ref) == 805
    assert {item['instruction'] for item in outputs} == set(ref)
    pairs = [(item['instruction'], unwrap(item['output']), ref[item['instruction']])
             for item in outputs]
    if output_dir is None:
        output_dir = root / 'results/autoj' / policy / 'reddit-2' / method
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance = {'policy': policy, 'method': method, 'count': 805,
                  'generation_sha256': hashlib.sha256(generation.read_bytes()).hexdigest(),
                  'reference_sha256': reference_sha,
                  'judge_model': os.environ.get('AUTOJ_MODEL', 'GAIR/autoj-13b'),
                  'temperature': 0, 'top_p': 1, 'max_tokens': 1024,
                  'context_length': int(os.environ.setdefault('AUTOJ_MAX_MODEL_LEN', '4096'))}
    (output_dir / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    llm = build_llm()
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
    summaries = {}
    for pos in (1, 2):
        rows = []
        with (output_dir / f'position-{pos}.jsonl').open('x') as stream:
            for start in range(0, len(pairs), 32):
                batch = pairs[start:start + 32]
                prompts = []
                for instruction, model, reference in batch:
                    a, b = (model, reference) if pos == 1 else (reference, model)
                    prompt = build_autoj_input(instruction, a, b, 'pairwise_tie')
                    assert len(tokenizer.encode(prompt)) + 1024 <= 4096
                    prompts.append(prompt)
                judgments = llm.generate(prompts, params)
                assert len(judgments) == len(batch)
                for offset, (pair, judgment) in enumerate(zip(batch, judgments)):
                    answer = judgment.outputs[0].text
                    label = extract_pariwise_result(answer)
                    row = {'index': start + offset, 'instruction': pair[0],
                           'model_position': pos, 'label': label,
                           'outcome': outcome(label, pos), 'judgment': answer}
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    rows.append(row)
                stream.flush()
                print(policy, method, 'position', pos, len(rows), flush=True)
        summaries[f'position{pos}'] = summarize(rows)
    summaries['mean_strict_win_rate'] = sum(
        summaries[f'position{pos}']['strict_win_rate'] for pos in (1, 2)) / 2
    summaries['mean_half_tie_win_rate'] = sum(
        summaries[f'position{pos}']['half_tie_win_rate'] for pos in (1, 2)) / 2
    (output_dir / 'result.json').write_text(json.dumps(summaries, indent=2))
    (output_dir / 'AUTOJ_COMPLETE').write_text('complete\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--root', type=Path, help='Legacy experiment directory')
    mode.add_argument('--generation-file', type=Path, help='Policy output JSON')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--policy', choices=('gemma', 'llama'), required=True)
    parser.add_argument('--method', choices=('fedbis', 'rand4', 'map_elites'), required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    if args.generation_file and args.output_dir is None:
        parser.error('--generation-file requires --output-dir')
    run(args.root, args.policy, args.method, args.reference,
        generation=args.generation_file, output_dir=args.output_dir)
