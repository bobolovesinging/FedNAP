"""
Some code snippets are borrowed from the open-sourced stanford_alpaca (
    https://github.com/tatsu-lab/stanford_alpaca)
"""

import copy
import logging
import math
import pandas as pd

from enum import Enum
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


def common_token_prefix_length(source, full):
    for index, (a, b) in enumerate(zip(source, full)):
        if a != b:
            return index
    return min(len(source), len(full))


# TODO: support LDA when 'category' in keys
class LLMDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_tag='output',
                 preserve_response_boundary=False):
        super(LLMDataset, self).__init__()
        self.preserve_response_boundary = preserve_response_boundary

        # Print prompt info
        logger.info(f'prompt_input: {prompt_input}')
        logger.info(f'prompt_no_input: {prompt_no_input}')

        self.sources = []
        for example in list_data_dict:
            input = example.get("input", None)
            if input is not None and input != "":
                self.sources.append(prompt_input.format_map(example))
            else:
                self.sources.append(prompt_no_input.format_map(example))

        targets = [
            f"{example[output_tag]}{tokenizer.eos_token}"
            for example in list_data_dict
        ]
        # self.llm_generated_targets = [
        #     f"{example['llm_output']}"
        #     if example.get("llm_output", "") != "" else None
        #     for example in list_data_dict
        # ]

        data_dict = self.preprocess(self.sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        self.tokenizer = tokenizer

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer):
        examples = [s + t for s, t in zip(sources, targets)]
        examples_tokenized, sources_tokenized = [
            self._tokenize_fn(strings, tokenizer)
            for strings in (examples, sources)
        ]
        input_ids = examples_tokenized["input_ids"]
        labels = copy.deepcopy(input_ids)
        for label, source_len, source_ids in zip(
                labels, sources_tokenized["input_ids_lens"],
                sources_tokenized["input_ids"]):
            if self.preserve_response_boundary:
                # SentencePiece may merge the last prompt token with the answer.
                source_len = common_token_prefix_length(source_ids, label)
            label[:source_len] = DefaultToken.IGNORE_INDEX.value
            # TODO: remove the data which is longer than the max input length
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])

    # def overwrite_by_llm(self, i):
    #     source = self.sources[i]
    #     llm_answer = self.llm_generated_targets[i]

    #     if llm_answer is None or llm_answer == "":
    #         return

    #     llm_result_tknz = \
    #         self.preprocess([source], [llm_answer], self.tokenizer)
    #     self.input_ids[i], self.labels[i] = \
    #         llm_result_tknz['input_ids'][0], llm_result_tknz['labels'][0]


class LLMComparisonDataset(Dataset):
    def __init__(self,
                 list_data_dict,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 output_A='output_A',
                 output_B='output_B',
                 choice='choice',
                 category_to_index=None,
                 preserve_response_boundary=False):
        # -- preserve original output_A/output_B (NO swap) ------------
        # d1+d2 augmentation: choice=0 -> output_A=winner, choice=1 -> output_B=winner
        # Win/lose selection is done in __getitem__ based on self.choices[i].
        filtered = []
        self.choices = []
        for example in list_data_dict:
            if choice not in example:
                continue
            self.choices.append(int(example[choice]))
            filtered.append(example)
        list_data_dict = filtered

        weight_presence = ['dpo_weight' in example
                           for example in list_data_dict]
        if any(weight_presence) and not all(weight_presence):
            raise ValueError(
                'dpo_weight must be present on all comparison samples or none')
        self.dpo_weights = None
        if weight_presence and weight_presence[0]:
            self.dpo_weights = [float(example['dpo_weight'])
                                for example in list_data_dict]
            if any(not math.isfinite(weight) or not 0.0 < weight <= 1.0
                   for weight in self.dpo_weights):
                raise ValueError(
                    'comparison dpo_weight values must be finite in (0, 1]')

        # dataset_A = tokenized output_A (may be winner or loser, depends on choice)
        # dataset_B = tokenized output_B (may be winner or loser, depends on choice)
        self.dataset_A = LLMDataset(list_data_dict=list_data_dict,
                                    tokenizer=tokenizer,
                                    prompt_input=prompt_input,
                                    prompt_no_input=prompt_no_input,
                                    output_tag=output_A,
                                    preserve_response_boundary=preserve_response_boundary)
        self.dataset_B = LLMDataset(list_data_dict=list_data_dict,
                                    tokenizer=tokenizer,
                                    prompt_input=prompt_input,
                                    prompt_no_input=prompt_no_input,
                                    output_tag=output_B,
                                    preserve_response_boundary=preserve_response_boundary)

        # ── bias labels (optional, used by FedCQD) ───────────────────────────
        # Detect presence of precomputed bias fields in data.
        # Labels are already in win/lose frame (precomputed relative to chosen).
        _BIAS_FIELDS = [
            # Cross-dataset spurious correlations (LLM-verified, both SHP and TLDR)
            # Sorted by combined LLM rate (SHP% + TLDR%):
            'bias_first_person',       # SHP 31%  / TLDR 41%  (LLM: 43%/70%)
            'bias_length',             # SHP 30%  / TLDR 35%  (LLM: 20%/24%)
            'bias_action_advice',      # SHP 27%  / TLDR 29%  (LLM: 36%/32%)
            'bias_punct',              # SHP 22%  / TLDR 23%  (LLM: 51%/77%)
            'bias_hedging',            # SHP 22%  / TLDR 13%  (LLM: 48%/42%)
            'bias_negative_framing',   # SHP 17%  / TLDR 19%  (LLM: 29%/55%)
            # Lower-signal factors (kept for completeness):
            'bias_specificity',        # SHP 18%  / TLDR 12%
            'bias_sentiment',          # SHP 23%  / TLDR 14%
            'bias_empathy',            # SHP  8%  / TLDR 10%  (low in both)
            'bias_lead',               # TLDR 43%  (2-gram precision vs post first 50 words)
            'bd_length_preference',     # chosen-vs-rejected normalized length margin
            'bd_lead_preference',       # chosen-vs-rejected lead-copy margin
            'bd_first_person_preference',  # chosen-vs-rejected first-person margin
        ]
        if list_data_dict and any(
                f in list_data_dict[0] for f in _BIAS_FIELDS):
            self.bias_labels = [
                {f: float(ex.get(f) or 0.0)
                 for f in _BIAS_FIELDS if f in ex}
                for ex in list_data_dict
            ]
        else:
            self.bias_labels = None

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        if category_to_index is None:
            df = pd.DataFrame(categories, columns=["category"])
            self.categories = list(pd.Categorical(df["category"]).codes)
            self.category_to_index = None
        else:
            missing = sorted({category for category in categories
                              if category not in category_to_index})
            if missing:
                raise ValueError(
                    'Missing categories in the fixed category mapping: '
                    f'{missing[:10]}')
            self.category_to_index = dict(category_to_index)
            self.categories = [self.category_to_index[category]
                               for category in categories]

        # super(LLMComparisonDataset, self).__init__(
        #     list_data_dict, tokenizer, prompt_input,
        #     prompt_no_input, output_A)

        # self.win_labels = self.labels

        # targets_B = [
        #     f"{example[output_B]}{tokenizer.eos_token}"
        #     for example in list_data_dict
        # ]
        # data_dict_B = self.preprocess(self.sources, targets_B, tokenizer)
        # self.lose_labels = data_dict_B["labels"]

    def __len__(self):
        return len(self.dataset_A)

    def __getitem__(self, i):
        if self.choices[i] == 0:
            # output_A is winner
            win_data = self.dataset_A[i]
            lose_data = self.dataset_B[i]
        else:
            # output_B is winner
            win_data = self.dataset_B[i]
            lose_data = self.dataset_A[i]
        item = dict(win_data=win_data,
                    lose_data=lose_data,
                    categories=self.categories[i])
        if self.bias_labels is not None:
            item['bias_labels'] = self.bias_labels[i]
        if self.dpo_weights is not None:
            item['dpo_weight'] = self.dpo_weights[i]
        return item
