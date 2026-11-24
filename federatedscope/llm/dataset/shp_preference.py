"""Deterministic SHP nuisance labels and client partition helpers."""

import copy
import json
import re
from collections import Counter, defaultdict

import numpy as np


_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_FIRST_PERSON = {
    'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours',
    'ourselves',
}
_FIRST_PERSON_CONTRACTIONS = {
    "i'm", "i've", "i'll", "i'd", "we're", "we've", "we'll", "we'd",
}
FIRST_PERSON_VERSION = 'contractions-v2'

SHP_COMPARISON_PROMPT = (
    'Below is a query followed by two responses. Pick a helpful response '
    'that is precise, concise, and casual. State your choice with a single '
    'capital letter, i.e., "A" if RESPONSE A is better, "B" if RESPONSE B '
    'is better.\n\n### QUERY: {instruction}\n### RESPONSE A: {output_A}'
    '\n### RESPONSE B: {output_B}\n### YOUR CHOICE:')

_PREPARED_FIELDS = (
    'instruction', 'output_A', 'output_B', 'choice', 'domain', 'category',
    'bias_length', 'bias_first_person', 'bd_length_preference',
    'bd_first_person_preference',
)


def read_prepared_shp_jsonl(path):
    """Read the strict Table VII schema without legacy input/output fields."""
    records = []
    with open(path, 'r', encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            item = json.loads(line)
            missing = [field for field in _PREPARED_FIELDS
                       if field not in item]
            if missing:
                raise ValueError(
                    f'prepared SHP line {line_number} is missing {missing}')
            records.append({field: item[field] for field in _PREPARED_FIELDS})
    if not records:
        raise ValueError(f'prepared SHP file is empty: {path}')
    return records


def select_prepared_shp_clients(records, expected_clients,
                                pool_all_clients=False):
    """Select a client prefix, or pool training clients for head pretraining."""
    client_count = int(expected_clients)
    if client_count <= 0:
        raise ValueError('expected_clients must be positive')
    if pool_all_clients:
        if client_count != 1:
            raise ValueError('Pooled pretraining requires one runtime client')
        if not records or any(
                not str(item['category']).startswith('Client_')
                or not str(item['category'])[7:].isdigit()
                for item in records):
            raise ValueError('Pooling requires nonempty training client records')
        return [dict(item, category='Client_0') for item in records]
    expected_names = [f'Client_{index}' for index in range(client_count)]
    observed_names = {item['category'] for item in records}
    missing = [name for name in expected_names if name not in observed_names]
    if missing:
        raise ValueError(f'Missing requested prepared clients: {missing}')
    selected_names = set(expected_names)
    return [item for item in records
            if item['category'] in selected_names]


def filter_shp_records_by_token_length(records, tokenizer, max_pair_tokens):
    """Apply the original SHP selector's combined 512-token train filter."""
    limit = int(max_pair_tokens)
    if limit <= 0:
        raise ValueError('max_pair_tokens must be positive')
    retained = []
    for item in records:
        total = sum(len(tokenizer(item[field])['input_ids'])
                    for field in ('instruction', 'output_A', 'output_B'))
        if total <= limit:
            retained.append(item)
    return retained


def validate_complete_shp_pairs(records, tokenizer, prompt):
    """Full-answer bias labels require complete prompt/response encodings."""
    limit = int(tokenizer.model_max_length)
    if not 0 < limit < 1_000_000:
        raise ValueError('Complete pair inputs require a finite context limit')
    for index, record in enumerate(records):
        source = prompt.format_map(record)
        for side in ('output_A', 'output_B'):
            encoded = tokenizer(source + record[side] + tokenizer.eos_token,
                                return_tensors='pt', truncation=False)
            length = encoded.input_ids.shape[-1]
            if length > limit:
                raise ValueError(
                    f'Cannot encode complete SHP pair: row={index} '
                    f'side={side} tokens={length} context_limit={limit}. '
                    'Full-answer labels must not be evaluated on truncated inputs.')


def align_shp_response_masks(records, pair_dataset, tokenizer):
    """Retain boundary-merged pieces using the original tokenizer's prefix."""
    for side, dataset in (('output_A', pair_dataset.dataset_A),
                          ('output_B', pair_dataset.dataset_B)):
        for index, record in enumerate(records):
            source, answer = dataset.sources[index], record[side]
            ids = tokenizer(source + answer + tokenizer.eos_token,
                            return_tensors='pt', truncation=False).input_ids[0]
            if not ids.equal(dataset.input_ids[index]):
                raise ValueError('SHP boundary audit does not match pair encoding')
            prefix = tokenizer(source, return_tensors='pt',
                               truncation=False).input_ids[0]
            common = min(len(prefix), len(ids))
            differences = (prefix[:common] != ids[:common]).nonzero()
            if len(differences):
                common = int(differences[0, 0])
            labels = ids.clone()
            labels[:common if answer.strip() else len(labels)] = -100
            if len(ids) and ids[-1].item() == tokenizer.eos_token_id:
                labels[-1] = ids[-1]
            dataset.labels[index] = labels


def validate_shp_response_masks(records, pair_dataset, tokenizer):
    """Reject lost answer tokens before training, while retaining empty answers."""
    for side, dataset in (('output_A', pair_dataset.dataset_A),
                          ('output_B', pair_dataset.dataset_B)):
        if len(dataset.input_ids) != len(records):
            raise ValueError('SHP response mask audit requires every original row')
        for index, (record, ids, labels) in enumerate(zip(
                records, dataset.input_ids, dataset.labels)):
            valid_eos = (len(ids) > 0 and ids[-1].item() == tokenizer.eos_token_id
                         and labels[-1].item() == tokenizer.eos_token_id)
            response_tokens = int((labels[:-1] != -100).sum())
            if not valid_eos or (record[side].strip() and response_tokens == 0):
                raise ValueError(
                    f'Invalid SHP response mask: row={index} side={side} '
                    f'characters={len(record[side])} response_tokens={response_tokens} '
                    f'complete_eos={valid_eos}')


def _tokens(text):
    return _WORD_RE.findall((text or '').lower())


def _chosen_rejected(item):
    choice = int(item['choice'])
    if choice not in (0, 1):
        raise ValueError(f'choice must be 0 or 1, got {choice!r}')
    outputs = (item['output_A'], item['output_B'])
    return outputs[choice], outputs[1 - choice]


def _first_person_rate(tokens):
    return (sum(token in _FIRST_PERSON for token in tokens)
            / max(len(tokens), 1))


def add_shp_causal_descriptors(record, length_ratio=1.25,
                               first_person_version='legacy-v1'):
    """Attach chosen-oriented binary labels and continuous BC margins."""
    item = copy.deepcopy(record)
    chosen, rejected = _chosen_rejected(item)
    chosen_tokens = _tokens(chosen)
    rejected_tokens = _tokens(rejected)
    if first_person_version == 'legacy-v1':
        chosen_first_person = _first_person_rate(chosen_tokens)
        rejected_first_person = _first_person_rate(rejected_tokens)
    elif first_person_version == FIRST_PERSON_VERSION:
        def rate(text):
            # Normalize apostrophes only for this factor; length stays legacy.
            tokens = _tokens((text or '').translate(str.maketrans(
                {'\u2019': "'", '\u2018': "'", '\u02bc': "'"})))
            words = _FIRST_PERSON | _FIRST_PERSON_CONTRACTIONS
            return sum(token in words for token in tokens) / max(len(tokens), 1)
        chosen_first_person, rejected_first_person = rate(chosen), rate(rejected)
    else:
        raise ValueError(f'Unknown first-person version: {first_person_version}')
    item['bias_length'] = float(
        bool(rejected_tokens)
        and len(chosen_tokens) / len(rejected_tokens) > length_ratio)
    item['bias_first_person'] = float(
        chosen_first_person > rejected_first_person)
    item['bd_length_preference'] = (
        (len(chosen_tokens) - len(rejected_tokens))
        / max(len(chosen_tokens) + len(rejected_tokens), 1))
    item['bd_first_person_preference'] = (
        chosen_first_person - rejected_first_person)
    return item


def assign_shp_prompts_to_clients(records, client_num, alpha, seed):
    """Materialize a deterministic domain-Dirichlet prompt partition."""
    client_num = int(client_num)
    if client_num <= 0 or float(alpha) <= 0:
        raise ValueError('client_num and alpha must be positive')
    prompt_domains = {}
    for item in records:
        prompt = str(item['instruction'])
        domain = str(item.get('domain', item.get('category', 'unknown')))
        # Match FedBiscuit's SHP loader: prompt deduplication keeps the
        # domain from the first occurrence in the pinned dataset order.
        prompt_domains.setdefault(prompt, domain)
    if len(prompt_domains) < client_num:
        raise ValueError('fewer unique prompts than clients')

    grouped = defaultdict(list)
    for prompt, domain in prompt_domains.items():
        grouped[domain].append(prompt)
    rng = np.random.default_rng(int(seed))
    assignments = {}
    client_counts = Counter()
    for domain in sorted(grouped):
        prompts = sorted(grouped[domain])
        rng.shuffle(prompts)
        probabilities = rng.dirichlet(
            np.full(client_num, float(alpha), dtype=float))
        counts = rng.multinomial(len(prompts), probabilities)
        offset = 0
        for client_id, count in enumerate(counts):
            for prompt in prompts[offset:offset + int(count)]:
                assignments[prompt] = client_id
                client_counts[client_id] += 1
            offset += int(count)

    empty_clients = [client_id for client_id in range(client_num)
                     if client_counts[client_id] == 0]
    for empty_client in empty_clients:
        donor = max(range(client_num), key=lambda cid: (client_counts[cid], -cid))
        if client_counts[donor] <= 1:
            raise RuntimeError('cannot rebalance SHP clients without emptying one')
        prompt = sorted(
            key for key, value in assignments.items() if value == donor)[-1]
        assignments[prompt] = empty_client
        client_counts[donor] -= 1
        client_counts[empty_client] += 1
    return dict(sorted(assignments.items()))


def prepare_shp_split(rows, split, client_assignments=None):
    """Convert raw SHP rows into the immutable Table VII JSONL schema."""
    split = str(split)
    prepared = []
    for row in rows:
        instruction = str(row['history'])
        domain = str(row['domain']).split('_')[0]
        if split == 'train':
            if client_assignments is None or instruction not in client_assignments:
                raise ValueError(
                    f'missing client assignment for prompt {instruction!r}')
            category = f'Client_{int(client_assignments[instruction])}'
        else:
            category = f'Eval_{split}_{domain}'
        item = {
            'instruction': instruction,
            'output_A': str(row['human_ref_A']),
            'output_B': str(row['human_ref_B']),
            'choice': 1 - int(row['labels']),
            'domain': domain,
            'category': category,
        }
        prepared.append(add_shp_causal_descriptors(item))
    return prepared
