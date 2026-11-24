import os
import json
import copy
import hashlib
import logging
import pickle
import random
import unicodedata

from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import DefaultToken, \
    LLMDataset, LLMComparisonDataset
from federatedscope.llm.dataset.tldr_preference import \
    add_causal_bias_labels, cap_records_per_category, \
    compute_preference_descriptors, format_worker_choice_records

logger = logging.getLogger(__name__)


class WorkerChoiceCausalDataset:
    """Choice examples paired with reusable winner/rejected encodings."""

    def __init__(self, choice_dataset, pair_dataset):
        self.choice_dataset = choice_dataset
        self.pair_dataset = pair_dataset
        self.categories = choice_dataset.categories
        self.category_to_index = getattr(
            choice_dataset, 'category_to_index', None)

    def __len__(self):
        return len(self.choice_dataset)

    def __getitem__(self, index):
        item = dict(self.choice_dataset[index])
        pair = self.pair_dataset[index % len(self.pair_dataset)]
        item.update(
            win_data=pair['win_data'],
            lose_data=pair['lose_data'],
            bias_labels=pair['bias_labels'],
        )
        return item

TLDR_PROMPT_DICT = {
    # "summary": ("Below is a forum post. Write a precise and concise summary "
    #             "that includes the most important points of the post.\n\n"
    #             "### SUBREDDIT: r/{subreddit}\n"
    #             "### TITLE: {title}\n"
    #             "### POST: {post}\n"
    #             "### TL;DR:"),
    "summary": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\nSummarize the following Reddit post in "
        "a paragraph of 50 words or less.\n\n"
        "### Input:\n"
        "SUBREDDIT: r/{subreddit}\n"
        "TITLE: {title}\n"
        "POST: {post}\n\n"
        "### Response:"),
    "summary_cmp": (
        "Below is a forum post followed by two summaries. "
        "Pick a more precise and concise one that summarizes the most "
        "important points in the given forum post, without including "
        "unimportant or irrelevant details. State your choice with a "
        "single capital letter, i.e., \"A\" if SUMMARY A is better, "
        "\"B\" if SUMMARY B is better.\n\n"
        "### SUBREDDIT: r/{subreddit}\n"
        "### TITLE: {title}\n"
        "### POST: {post}\n"
        "### SUMMARY A:{output_A}\n"
        "### SUMMARY B:{output_B}\n"
        "### YOUR CHOICE:"),
    "mix_cmp": ("Below is an instruction that describes a task, "
                "paired with an input that provides further context. "
                "There are two responses that complete the request. "
                "Pick an appropriate response and state your choice with "
                "a single capital letter, i.e., "
                "\"A\" if RESPONSE A is better and more appropriate, "
                "\"B\" if RESPONSE B is better and more appropriate.\n\n"
                "### Instruction:\nSummarize the following Reddit post.\n\n"
                "### Input:\n"
                "SUBREDDIT: r/{subreddit}\n"
                "TITLE: {title}\n"
                "POST: {post}\n\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:")
}


def _download_tldr_cmpr(data_root):
    all_files = [f'batch{i}' for i in range(3, 21)] + ['batch22']
    # all_files = ['batch8']

    for cmp_file in all_files:
        download_url(
            'https://openaipublic.blob.core.windows.net/'
            'summarize-from-feedback/dataset/comparisons/'
            f'{cmp_file}.json', data_root)

    # Preprocess the above data
    file_paths = [
        os.path.join(data_root, f'{cmp_file}.json') for cmp_file in all_files
    ]
    list_data_dict = load_jsonls(file_paths,
                                 subreddit='info.subreddit',
                                 title='info.title',
                                 post='info.post',
                                 output_A='summaries.0.text',
                                 output_B='summaries.1.text',
                                 category='worker',
                                 split='split',
                                 choice='choice')
    # Split the dataset into ['train', 'val', 'test']
    list_train_dict, list_val_dict, list_test_dict = [], [], []
    for data_dict in list_data_dict:
        if data_dict['split'] == 'train':
            list_train_dict.append(data_dict)
        elif data_dict['split'] == 'valid1':
            list_val_dict.append(data_dict)
        elif data_dict['split'] == 'valid2':
            list_test_dict.append(data_dict)

    # # merge the worker with less than 10 samples
    # dict_cat = {}
    # for idx, sample in enumerate(list_train_dict):
    #     if sample['category'] not in dict_cat:
    #         dict_cat[sample['category']] = [idx]
    #     else:
    #         dict_cat[sample['category']].append(idx)
    # for values in dict_cat.values():
    #     if len(values) < 50:
    #         for idx in values:
    #             list_train_dict[idx]['category'] = 'merged_client'

    return list_train_dict, list_val_dict, list_test_dict


def _download_tldr_human(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'reddit-tldr_train.jsonl'),
        os.path.join(data_root, 'reddit-tldr_val.jsonl'),
        os.path.join(data_root, 'reddit-tldr_test.jsonl')
    ]

    for name, fp in [('train', train_fp), ('valid', val_fp),
                     ('test', test_fp)]:
        if not os.path.exists(fp):
            download_url(
                'https://openaipublic.blob.core.windows.net/'
                'summarize-from-feedback/datasets/'
                f'tldr_3_filtered/{name}.jsonl', data_root)
            os.rename(os.path.join(data_root, f'{name}.jsonl'), fp)

    dataloader_kwargs = {
        'subreddit': 'subreddit',
        'title': 'title',
        'post': 'post',
        'summary': 'summary'
    }
    list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
    list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
    list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    return list_train_dict, list_val_dict, list_test_dict


def get_tldr_dataset(list_data_dict,
                     tokenizer,
                     prompt=TLDR_PROMPT_DICT['summary']):
    return LLMDataset(list_data_dict,
                      tokenizer,
                      prompt_input=prompt,
                      prompt_no_input=prompt,
                      output_tag='summary')


def load_human_annotated_dataset(data_root, tokenizer):
    list_train_dict, list_val_dict, list_test_dict = \
        _download_tldr_human(data_root)

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_human_finetuning_dataset(data_root,
                                  tokenizer,
                                  rlhf=False,
                                  max_num_test=-1,
                                  raw_no_prompt=False):
    staged_names = ('sft.jsonl', 'candidates.jsonl', 'policy_val.jsonl',
                    'policy_dev_holdout.jsonl')
    missing = [name for name in staged_names
               if not os.path.isfile(os.path.join(data_root, name))]
    if missing:
        raise FileNotFoundError(
            f"Missing staged TLDR files in {data_root}: {', '.join(missing)}. "
            "Run scripts/stage_tldr.py using the official inputs.")

    def read_staged(name):
        with open(os.path.join(data_root, name), encoding='utf-8') as stream:
            return [json.loads(line) for line in stream if line.strip()]

    list_train_dict = read_staged('candidates.jsonl' if rlhf else 'sft.jsonl')
    # The runner initializes a dataset with a target field before loading
    # generated preferences for RLHF training.
    if rlhf and not raw_no_prompt:
        list_train_dict = [dict(row, summary='') for row in list_train_dict]
    list_val_dict = read_staged('policy_val.jsonl')
    list_test_dict = read_staged('policy_dev_holdout.jsonl')
    if raw_no_prompt:
        if max_num_test > 0:
            return (list_train_dict, list_val_dict[:max_num_test],
                    list_test_dict[:max_num_test])
        else:
            return list_train_dict, list_val_dict, list_test_dict

    train_dataset = LLMDataset(list_train_dict,
                               tokenizer,
                               prompt_input=TLDR_PROMPT_DICT['summary'],
                               prompt_no_input=TLDR_PROMPT_DICT['summary'],
                               output_tag='summary')
    val_dataset = LLMDataset(list_val_dict,
                             tokenizer,
                             prompt_input=TLDR_PROMPT_DICT['summary'],
                             prompt_no_input=TLDR_PROMPT_DICT['summary'],
                             output_tag='summary')
    test_dataset = LLMDataset(list_test_dict,
                              tokenizer,
                              prompt_input=TLDR_PROMPT_DICT['summary'],
                              prompt_no_input=TLDR_PROMPT_DICT['summary'],
                              output_tag='summary')

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_comparison_dataset(data_root, tokenizer, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    train_set_path = os.path.join(data_root, f'{token_name}_train.pickle')
    val_set_path = os.path.join(data_root, f'{token_name}_val.pickle')
    test_set_path = os.path.join(data_root, f'{token_name}_test.pickle')
    if os.path.exists(train_set_path) and os.path.exists(val_set_path) \
            and os.path.exists(test_set_path):
        with open(train_set_path, 'rb') as f_train, \
                open(val_set_path, 'rb') as f_val, \
                open(test_set_path, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

        # load dataset, which should be tuple
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(train_set_path, 'wb') as f_train, \
                open(val_set_path, 'wb') as f_val, \
                open(test_set_path, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.win_dataset.input_ids = \
            val_dataset.win_dataset.input_ids[:max_num_test]
        val_dataset.lose_dataset.input_ids = \
            val_dataset.lose_dataset.input_ids[:max_num_test]
        test_dataset.win_dataset.input_ids = \
            test_dataset.win_dataset.input_ids[:max_num_test]
        test_dataset.lose_dataset.input_ids = \
            test_dataset.lose_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_best_dataset(data_root, tokenizer, max_num_test=-1):
    train_dataset, val_dataset, test_dataset = \
        load_comparison_dataset(data_root, tokenizer, max_num_test)
    # Use the win_dataset only
    dataset = (train_dataset.win_dataset, val_dataset.win_dataset,
               test_dataset.win_dataset)
    return dataset


def load_comparison_dataset_by_choice(data_root, tokenizer, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    train_set_path = os.path.join(data_root,
                                  f'{token_name}_train_choice.pickle')
    val_set_path = os.path.join(data_root, f'{token_name}_val_choice.pickle')
    test_set_path = os.path.join(data_root, f'{token_name}_test_choice.pickle')
    if os.path.exists(train_set_path) and os.path.exists(val_set_path) and \
            os.path.exists(test_set_path):
        with open(train_set_path, 'rb') as f_train, \
                open(val_set_path, 'rb') as f_val, \
                open(test_set_path, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            _download_tldr_cmpr(data_root)

        # For training dataset, we should exchange the order
        # and append the new training dataset to the list_train_dict
        exchange_list_train_dict = copy.deepcopy(list_train_dict)
        for sample in exchange_list_train_dict:
            sample['output_A'], sample['output_B'] = \
                sample['output_B'], sample['output_A']
            sample['choice'] = 1 - sample['choice']
        list_train_dict = list_train_dict + exchange_list_train_dict

        # map the choice to "A" and "B" instead of 0 and 1
        for list_dict in [list_train_dict, list_test_dict, list_val_dict]:
            for sample in list_dict:
                sample['choice'] = " " + chr(sample['choice'] + ord("A"))

        train_dataset = LLMDataset(
            list_train_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')
        val_dataset = LLMDataset(
            list_val_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')
        test_dataset = LLMDataset(
            list_test_dict,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
            prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
            output_tag='choice')

        # Store these three lists to a pickle file
        with open(train_set_path, 'wb') as f_train, \
                open(val_set_path, 'wb') as f_val, \
                open(test_set_path, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def check_sim(data_root):
    cmpr_list_train_dict, cmpr_list_val_dict, cmpr_list_test_dict = \
        _download_tldr_cmpr(data_root)

    human_list_train_dict, human_list_val_dict, human_list_test_dict = \
        _download_tldr_human(data_root)

    # show if human-annotated overlaps cmpr in terms of train_dict
    cmpr_train = [sample['post'] for sample in cmpr_list_val_dict]
    human_train = [sample['post'] for sample in human_list_train_dict]

    print(len(cmpr_train))  # 92858
    print(len(human_train))  # 116722

    total_overlapping = 0

    for data in cmpr_train:
        if data in human_train:
            total_overlapping += 1
            human_train.pop(human_train.index(data))

    print(len(human_train))
    print(total_overlapping)  # 59685/282/475


def load_comparison_dataset_with_bias(data_root, tokenizer, max_num_test=-1):
    """
    Loads bias-enriched TLDR comparison files
    (tldr_cmpr_{train,val,test}_bias.jsonl) produced by precompute_bias_labels.py.

    Returns LLMComparisonDataset objects whose .bias_labels field is populated,
    so LLMRewardCollator will pass bias_* tensors in every batch.
    """
    token_name    = os.path.basename(tokenizer.name_or_path)
    train_fp_pkl  = os.path.join(data_root, f'{token_name}_train_bias.pickle')
    val_fp_pkl    = os.path.join(data_root, f'{token_name}_val_bias.pickle')
    test_fp_pkl   = os.path.join(data_root, f'{token_name}_test_bias.pickle')

    if (os.path.exists(train_fp_pkl) and os.path.exists(val_fp_pkl)
            and os.path.exists(test_fp_pkl)):
        with open(train_fp_pkl, 'rb') as ft, \
             open(val_fp_pkl,   'rb') as fv, \
             open(test_fp_pkl,  'rb') as fx:
            train_dataset = pickle.load(ft)
            val_dataset   = pickle.load(fv)
            test_dataset  = pickle.load(fx)
    else:
        train_raw = os.path.join(data_root, 'tldr_cmpr_train_bias.jsonl')
        val_raw   = os.path.join(data_root, 'tldr_cmpr_val_bias.jsonl')
        test_raw  = os.path.join(data_root, 'tldr_cmpr_test_bias.jsonl')

        if not os.path.exists(train_raw):
            import warnings
            warnings.warn(
                "TLDR bias files not found. "
                "Run precompute_bias_labels.py first. "
                "Falling back to plain comparison dataset.")
            return load_comparison_dataset(data_root, tokenizer, max_num_test)

        bias_kwargs = {
            'post'                 : 'post',
            'subreddit'            : 'subreddit',
            'title'                : 'title',
            'output_A'             : 'output_A',
            'output_B'             : 'output_B',
            'choice'               : 'choice',
            'category'             : 'category',
            'bias_length'          : 'bias_length',
            'bias_first_person'    : 'bias_first_person',
            'bias_action_advice'   : 'bias_action_advice',
            'bias_punct'           : 'bias_punct',
            'bias_negative_framing': 'bias_negative_framing',
            'bias_sentiment'       : 'bias_sentiment',
        }

        from federatedscope.llm.dataloader.dataloader import load_jsonl
        list_train = load_jsonl(train_raw, **bias_kwargs)
        list_val   = load_jsonl(val_raw,   **bias_kwargs)
        list_test  = load_jsonl(test_raw,  **bias_kwargs)

        def _make(lst):
            return LLMComparisonDataset(
                lst, tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary'],
                prompt_no_input=TLDR_PROMPT_DICT['summary'],
                output_A='output_A',
                output_B='output_B',
                choice='choice')

        train_dataset = _make(list_train)
        val_dataset   = _make(list_val)
        test_dataset  = _make(list_test)

        with open(train_fp_pkl, 'wb') as ft, \
             open(val_fp_pkl,   'wb') as fv, \
             open(test_fp_pkl,  'wb') as fx:
            pickle.dump(train_dataset, ft)
            pickle.dump(val_dataset,   fv)
            pickle.dump(test_dataset,  fx)

    if max_num_test > 0:
        for ds in (val_dataset, test_dataset):
            ds.win_dataset.input_ids  = ds.win_dataset.input_ids[:max_num_test]
            ds.lose_dataset.input_ids = ds.lose_dataset.input_ids[:max_num_test]
            if ds.bias_labels is not None:
                ds.bias_labels = ds.bias_labels[:max_num_test]

    return train_dataset, val_dataset, test_dataset


def load_worker_partitioned_tldr(data_root,
                                 tokenizer,
                                 worker_num=53,
                                 max_train=-1,
                                 max_train_per_worker=-1,
                                 public_eval_size=0,
                                 test_eval_size=0,
                                 choice_format=False,
                                 causal_choice=False):
    """Load bias-enriched TLDR comparisons with stable worker ownership.

    Training annotators define clients ``0..worker_num-1``. Annotators that
    appear only in validation or test receive IDs above that range; the strict
    metadata splitter keeps those examples in the server dataset and excludes
    them from known-client validation/test partitions.
    """
    filenames = {
        'train': 'tldr_cmpr_train_bias.jsonl',
        'val': 'tldr_cmpr_val_bias.jsonl',
        'test': 'tldr_cmpr_test_bias.jsonl',
    }
    paths = {split: os.path.join(data_root, filename)
             for split, filename in filenames.items()}
    missing_paths = [path for path in paths.values()
                     if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Missing worker-partitioned TLDR files: '
            + ', '.join(missing_paths))

    fields = {
        'post': 'post',
        'subreddit': 'subreddit',
        'title': 'title',
        'output_A': 'output_A',
        'output_B': 'output_B',
        'choice': 'choice',
        'category': 'category',
        'bias_length': 'bias_length',
        'bias_first_person': 'bias_first_person',
        'bias_action_advice': 'bias_action_advice',
        'bias_punct': 'bias_punct',
        'bias_negative_framing': 'bias_negative_framing',
        'bias_sentiment': 'bias_sentiment',
    }
    from federatedscope.llm.dataloader.dataloader import load_jsonl
    records = {split: load_jsonl(path, **fields)
               for split, path in paths.items()}
    for split_records in records.values():
        for item in split_records:
            length_margin, lead_margin = compute_preference_descriptors(item)
            item['bd_length_preference'] = length_margin
            item['bd_lead_preference'] = lead_margin

    train_workers = sorted({item['category'] for item in records['train']})
    if len(train_workers) != worker_num:
        raise ValueError(
            f'Expected exactly {worker_num} training workers, found '
            f'{len(train_workers)} in {paths["train"]}.')
    eval_workers = sorted(
        ({item['category'] for item in records['val']}
         | {item['category'] for item in records['test']})
        - set(train_workers))
    worker_to_index = {
        worker: index
        for index, worker in enumerate(train_workers + eval_workers)
    }

    if max_train_per_worker > 0:
        records['train'] = cap_records_per_category(
            records['train'], max_train_per_worker)
        retained_workers = {item['category'] for item in records['train']}
        if retained_workers != set(train_workers):
            raise RuntimeError(
                'Per-worker cap unexpectedly removed a training worker')

    if max_train > 0 and max_train < len(records['train']):
        if max_train < worker_num:
            raise ValueError(
                f'max_train={max_train} cannot retain all {worker_num} '
                'training workers.')
        first_by_worker = {}
        for index, item in enumerate(records['train']):
            first_by_worker.setdefault(item['category'], index)
        selected = set(first_by_worker.values())
        for index in range(len(records['train'])):
            if len(selected) >= max_train:
                break
            selected.add(index)
        records['train'] = [item for index, item in enumerate(records['train'])
                            if index in selected]

    def _panel_key(item):
        identity = json.dumps([
            item.get('category'), item.get('post'), item.get('title'),
            item.get('output_A'), item.get('output_B'), item.get('choice')
        ], ensure_ascii=True, separators=(',', ':'))
        return hashlib.sha256(identity.encode('utf-8')).digest()

    if public_eval_size > 0:
        records['val'] = sorted(records['val'], key=_panel_key)[
            :public_eval_size]
    if test_eval_size > 0:
        records['test'] = sorted(records['test'], key=_panel_key)[
            :test_eval_size]

    def _make(items, augment_choice=False):
        if choice_format:
            pair_items = (add_causal_bias_labels(items)
                          if causal_choice else list(items))
            choice_items = format_worker_choice_records(
                pair_items, augment=augment_choice)
            choice_dataset = LLMDataset(
                choice_items,
                tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary_cmp'],
                prompt_no_input=TLDR_PROMPT_DICT['summary_cmp'],
                output_tag='choice')
            choice_dataset.categories = [
                worker_to_index[item['category']] for item in choice_items
            ]
            choice_dataset.category_to_index = dict(worker_to_index)
            if not causal_choice:
                return choice_dataset
            pair_dataset = LLMComparisonDataset(
                pair_items,
                tokenizer,
                prompt_input=TLDR_PROMPT_DICT['summary'],
                prompt_no_input=TLDR_PROMPT_DICT['summary'],
                output_A='output_A',
                output_B='output_B',
                choice='choice',
                category_to_index=worker_to_index)
            return WorkerChoiceCausalDataset(
                choice_dataset=choice_dataset,
                pair_dataset=pair_dataset)
        return LLMComparisonDataset(
            items,
            tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice',
            category_to_index=worker_to_index)

    logger.info(
        'Loaded worker-partitioned TLDR: %d training workers, %d unseen '
        'evaluation workers; records train=%d val=%d test=%d.',
        len(train_workers), len(eval_workers), *(len(records[split])
                                                 for split in ('train',
                                                               'val',
                                                               'test')))
    datasets = (
        _make(records['train'], augment_choice=choice_format),
        _make(records['val']),
        _make(records['test']),
    )
    if public_eval_size > 0:
        datasets[1].replicate_to_clients = True
        datasets[1].public_panel_size = len(records['val'])
    return datasets


def load_llm_labeled_tldr(json_path, tokenizer, config=None, max_train=None):
    """Split annotated TLDR posts before producing both response orders."""
    field_map = {
        'first_person': 'bias_first_person',
        'length': 'bias_length',
        'action_advice': 'bias_action_advice',
        'punct': 'bias_punct',
        'neg_framing': 'bias_negative_framing',
        'lead': 'bias_lead',
    }
    category_mode = getattr(getattr(config, 'data', None), 'category_mode',
                            'subreddit')
    print(f"[FedCQD-TLDR] category_mode={category_mode}")

    with open(json_path, encoding='utf-8') as f:
        raw = json.load(f)

    groups_by_post = {}
    for item in raw:
        post = item['data']['post']
        key = ' '.join(unicodedata.normalize('NFKC', post).casefold().split())
        if not key:
            raise ValueError("Expected a nonempty post in the TLDR annotation file")
        groups_by_post.setdefault(key, []).append(item)

    rng = random.Random(42)
    groups = list(groups_by_post.values())
    rng.shuffle(groups)
    train_end = int(len(groups) * 0.8)
    val_end = train_end + int(len(groups) * 0.1)

    def _make(split_groups):
        records = []
        for group in split_groups:
            for item in group:
                base = dict(item['data'])
                labels = item['labels']
                chosen = base.pop('chosen', base.get('output_A'))
                rejected = base.pop('rejected', base.get('output_B'))
                base.setdefault('title', '')

                category = base.get('category', base.get('subreddit', 'unknown'))
                if category_mode == 'bias':
                    lead = labels.get('lead', 'no')
                    length = labels.get('length', 'no')
                    category = f"lead_{lead}_len_{length}"
                base['category'] = category
                base.update({
                    target: float(str(labels.get(source, 'no')).lower() == 'yes')
                    for source, target in field_map.items()
                })
                records.append(dict(base, output_A=chosen, output_B=rejected,
                                    choice=0))
                records.append(dict(base, output_A=rejected, output_B=chosen,
                                    choice=1))

        rng.shuffle(records)
        return LLMComparisonDataset(
            records, tokenizer,
            prompt_input=TLDR_PROMPT_DICT['summary'],
            prompt_no_input=TLDR_PROMPT_DICT['summary'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

    if any('split' in item for item in raw):
        splits = {name: [] for name in ('train', 'val', 'test')}
        for group in groups:
            names = {item.get('split') for item in group}
            if len(names) != 1 or next(iter(names)) not in splits:
                raise ValueError('Inconsistent staged head split')
            splits[names.pop()].append(group)
        train, val, test = (_make(splits[name]) for name in ('train', 'val', 'test'))
    else:
        train = _make(groups[:train_end])
        val = _make(groups[train_end:val_end])
        test = _make(groups[val_end:])
    print(f"[FedCQD-TLDR] Loaded {len(raw)} comparisons from {len(groups)} "
          f"distinct posts, with {len(train)}/{len(val)}/{len(test)} "
          "augmented train/validation/test records")
    return train, val, test


if __name__ == "__main__":
    data_root = os.path.join('/local/scratch/d/wu1977/dataset/',
                             'reddit-tldr-comparison')
    check_sim(data_root)
