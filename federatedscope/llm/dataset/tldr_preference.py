import re
import copy
from collections import Counter


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(text):
    return _WORD_RE.findall((text or '').lower())


def _lead_bigram_precision(summary, post, lead_words=50):
    lead = _tokens(post)[:lead_words]
    summary_tokens = _tokens(summary)
    if len(summary_tokens) < 2:
        return 0.0
    lead_bigrams = Counter(zip(lead, lead[1:]))
    summary_bigrams = Counter(zip(summary_tokens, summary_tokens[1:]))
    overlap = sum(min(count, lead_bigrams[bigram])
                  for bigram, count in summary_bigrams.items())
    return overlap / sum(summary_bigrams.values())


def compute_preference_descriptors(item):
    """Return orientation-invariant length and lead-copy preference margins."""
    output_a = item['output_A']
    output_b = item['output_B']
    if int(item['choice']) == 0:
        chosen, rejected = output_a, output_b
    else:
        chosen, rejected = output_b, output_a

    chosen_len = len(_tokens(chosen))
    rejected_len = len(_tokens(rejected))
    length_margin = ((chosen_len - rejected_len)
                     / max(chosen_len + rejected_len, 1))
    lead_margin = (
        _lead_bigram_precision(chosen, item['post'])
        - _lead_bigram_precision(rejected, item['post']))
    return length_margin, lead_margin


def compute_causal_bias_labels(item, length_ratio=1.25, lead_words=50,
                               summary_words=10):
    """Compute binary bias labels in the chosen/rejected orientation."""
    output_a = item['output_A']
    output_b = item['output_B']
    if int(item['choice']) == 0:
        chosen, rejected = output_a, output_b
    else:
        chosen, rejected = output_b, output_a

    chosen_tokens = _tokens(chosen)
    rejected_tokens = _tokens(rejected)
    bias_length = float(
        bool(rejected_tokens)
        and len(chosen_tokens) / len(rejected_tokens) > length_ratio)

    lead_tokens = set(_tokens(item.get('post', ''))[:lead_words])
    chosen_lead = sum(
        token in lead_tokens for token in chosen_tokens[:summary_words])
    rejected_lead = sum(
        token in lead_tokens for token in rejected_tokens[:summary_words])
    bias_lead = float(chosen_lead > rejected_lead)
    return {
        'bias_length': bias_length,
        'bias_lead': bias_lead,
    }


def add_causal_bias_labels(records):
    """Copy records and attach chosen-oriented causal bias labels."""
    labeled = []
    for record in records:
        item = copy.deepcopy(record)
        item.update(compute_causal_bias_labels(item))
        labeled.append(item)
    return labeled


def cap_records_per_category(records, limit):
    """Keep at most ``limit`` records per category in input order."""
    if limit <= 0:
        return list(records)
    retained = Counter()
    output = []
    for item in records:
        category = item['category']
        if retained[category] >= limit:
            continue
        output.append(item)
        retained[category] += 1
    return output


def format_worker_choice_records(records, augment=False):
    """Convert comparison records to FedBiscuit A/B choice examples."""
    originals = copy.deepcopy(list(records))
    formatted = []
    for item in originals:
        choice = int(item['choice'])
        if choice not in (0, 1):
            raise ValueError(f'choice must be 0 or 1, got {choice!r}')
        item['choice'] = f' {chr(ord("A") + choice)}'
        formatted.append(item)

    if augment:
        for source in records:
            item = copy.deepcopy(source)
            choice = int(item['choice'])
            if choice not in (0, 1):
                raise ValueError(f'choice must be 0 or 1, got {choice!r}')
            item['output_A'], item['output_B'] = (
                item['output_B'], item['output_A'])
            item['choice'] = f' {chr(ord("A") + 1 - choice)}'
            formatted.append(item)
    return formatted
