import os
import json
import copy
import pickle
import random
import logging
import datasets

logger = logging.getLogger(__name__)

from federatedscope.core.splitters.generic.lda_splitter import LDASplitter
from federatedscope.core.data.utils import download_url
from federatedscope.llm.dataloader.dataloader import load_jsonls, load_jsonl
from federatedscope.llm.dataset.llm_dataset import LLMComparisonDataset, \
    LLMDataset

SHP_PROMPT_DICT = {
    "shp": ("Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n"
            "### Response:"),
    "shp_cmp": ("Below is a query followed by two responses. Pick a "
                "helpful response that is precise, concise, and casual. "
                "State your choice with a single capital letter, "
                "i.e., \"A\" if RESPONSE A is better, "
                "\"B\" if RESPONSE B is better.\n\n"
                "### QUERY: {instruction}\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:"),
    "mix_cmp": ("Below is an instruction that describes a task. "
                "There are two responses that complete the request. "
                "Pick an appropriate response and state your choice with "
                "a single capital letter, i.e., "
                "\"A\" if RESPONSE A is better and more appropriate, "
                "\"B\" if RESPONSE B is better and more appropriate.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### RESPONSE A: {output_A}\n"
                "### RESPONSE B: {output_B}\n"
                "### YOUR CHOICE:")
}


def _download_shp_cmpr(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'shp_cmpr_train.jsonl'),
        os.path.join(data_root, 'shp_cmpr_val.jsonl'),
        os.path.join(data_root, 'shp_cmpr_test.jsonl')
    ]

    dataloader_kwargs = {
        'instruction': 'instruction',
        'output_A': 'output_A',
        'output_B': 'output_B',
        'choice': 'choice',
        'category': 'category'
    }
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        dataset = datasets.load_dataset("stanfordnlp/SHP")
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        tag_fp = {
            'train': (train_fp, list_train_dict),
            'validation': (val_fp, list_val_dict),
            'test': (test_fp, list_test_dict)
        }
        for tag, (fp, list_data_dict) in tag_fp.items():
            file = open(fp, 'w')
            for hist, ref_A, ref_B, choice, domain in \
                zip(dataset[tag]['history'],
                    dataset[tag]['human_ref_A'],
                    dataset[tag]['human_ref_B'],
                    dataset[tag]['labels'],
                    dataset[tag]['domain']):
                record = {
                    'instruction': hist,
                    'output_A': ref_A,
                    'output_B': ref_B,
                    'choice': choice,
                    'category': domain.split('_')[0]
                }
                file.write(f'{json.dumps(record)}\n')
                list_data_dict.append(record)
            file.close()

    return list_train_dict, list_val_dict, list_test_dict


def _download_shp(data_root):
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, 'shp_rlhf_train.jsonl'),
        os.path.join(data_root, 'shp_rlhf_val.jsonl'),
        os.path.join(data_root, 'shp_rlhf_test.jsonl')
    ]

    dataloader_kwargs = {'instruction': 'instruction', 'category': 'category'}
    if os.path.exists(train_fp) and os.path.exists(val_fp) and \
            os.path.exists(test_fp):
        list_train_dict = load_jsonl(train_fp, **dataloader_kwargs)
        list_val_dict = load_jsonl(val_fp, **dataloader_kwargs)
        list_test_dict = load_jsonl(test_fp, **dataloader_kwargs)

    else:
        dataset = datasets.load_dataset("stanfordnlp/SHP")
        instructions = []
        list_train_dict, list_val_dict, list_test_dict = [], [], []
        tag_fp = {
            'train': (train_fp, list_train_dict),
            'validation': (val_fp, list_val_dict),
            'test': (test_fp, list_test_dict)
        }
        for tag, (fp, list_data_dict) in tag_fp.items():
            file = open(fp, 'w')
            for hist, domain in zip(dataset[tag]['history'],
                                    dataset[tag]['domain']):
                if hist not in instructions:
                    instructions.append(hist)
                    record = {
                        'instruction': hist,
                        'category': domain.split('_')[0]
                    }
                    file.write(f'{json.dumps(record)}\n')
                    list_data_dict.append(record)
            file.close()

    return list_train_dict, list_val_dict, list_test_dict


def shp_dataset(data_root, num_clients, tokenizer, max_len=512):
    list_train_dict, list_val_dict, list_test_dict = \
        _download_shp_cmpr(data_root)

    # First, disjoint by post instructions
    list_train_instructions, _, _ = _download_shp(data_root)
    cat_idx_map = {}
    for sample in list_train_instructions:
        if sample['category'] not in cat_idx_map:
            cat_idx_map[sample['category']] = len(cat_idx_map)
        sample['categories'] = cat_idx_map[sample['category']]

    # Second, use Dirichlet data splitter
    splitter = LDASplitter(num_clients, alpha=0.3)
    inst_split_list = splitter(list_train_instructions)
    inst_client_map = {}
    for idx, sublist in enumerate(inst_split_list):
        for sample in sublist:
            inst_client_map[sample['instruction']] = idx

    # Update their categories and force the data splitter as meta
    for sample in list_train_dict:
        sample['domain'] = sample['category']
        sample['category'] = \
            f"Client_{inst_client_map[sample['instruction']]}"

    # Select the data with less or equal to 512 tokens
    new_list_train_dict = []
    for sample in list_train_dict:
        len_inst = len(tokenizer(sample['instruction'])['input_ids'])
        len_resA = len(tokenizer(sample['output_A'])['input_ids'])
        len_resB = len(tokenizer(sample['output_B'])['input_ids'])
        if len_inst + len_resA + len_resB <= max_len:
            new_list_train_dict.append(sample)
    list_train_dict = new_list_train_dict

    # Print the samples of each domain for each clients
    for client_id in range(num_clients + 1):
        print(f'Client {client_id}:')
        num_sample_by_domains = dict()
        for sample in new_list_train_dict:
            if sample['category'] == f'Client_{client_id}':
                if sample['domain'] not in num_sample_by_domains:
                    num_sample_by_domains[sample['domain']] = 0
                num_sample_by_domains[sample['domain']] += 1
        print(num_sample_by_domains)

    return list_train_dict, list_val_dict, list_test_dict


def load_rlhf_dataset(data_root,
                      tokenizer,
                      max_num_test=-1,
                      raw_no_prompt=False):
    _, list_val_dict, list_test_dict = \
        _download_shp(data_root)

    # reorganize the training data for RLHF
    list_train_dict = list_val_dict + list_test_dict
    list_val_dict = list_test_dict[:len(list_test_dict) // 2]
    list_test_dict = list_test_dict[len(list_test_dict) // 2:]

    if max_num_test > 0:
        return (list_train_dict, list_val_dict[:max_num_test],
                list_test_dict[:max_num_test])
    else:
        return list_train_dict, list_val_dict, list_test_dict


def load_safe_dataset():
    ds = datasets.load_dataset("PKU-Alignment/PKU-SafeRLHF-prompt")
    list_train_dict = [{
        'instruction': prompt
    } for prompt in ds['train']['prompt']]

    return list_train_dict, None, None


def load_comparison_dataset(data_root, tokenizer, config, max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    num_clients = config.federate.client_num
    train_fp, val_fp, test_fp = [
        os.path.join(data_root, f'{token_name}_train_{num_clients}.pickle'),
        os.path.join(data_root, f'{token_name}_val.pickle'),
        os.path.join(data_root, f'{token_name}_test.pickle')
    ]

    if os.path.exists(train_fp) and os.path.exists(val_fp) and os.path.exists(
            test_fp):
        with open(train_fp, 'rb') as f_train, open(val_fp, 'rb') as f_val, \
                open(test_fp, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            shp_dataset(data_root, num_clients, tokenizer,
                        max_len=config.llm.tok_len)

        # load dataset, which should be tuple
        train_dataset = LLMComparisonDataset(
            list_train_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        val_dataset = LLMComparisonDataset(
            list_val_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')
        test_dataset = LLMComparisonDataset(
            list_test_dict,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice')

        # Store these three lists to a pickle file
        with open(train_fp, 'wb') as f_train, \
                open(val_fp, 'wb') as f_val, \
                open(test_fp, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.dataset_A.input_ids = \
            val_dataset.dataset_A.input_ids[:max_num_test]
        val_dataset.dataset_B.input_ids = \
            val_dataset.dataset_B.input_ids[:max_num_test]
        test_dataset.dataset_A.input_ids = \
            test_dataset.dataset_A.input_ids[:max_num_test]
        test_dataset.dataset_B.input_ids = \
            test_dataset.dataset_B.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_prepared_shp_table_vii(data_root, tokenizer, config,
                                choice_format=True):
    cache_root = os.environ.get("FEDCQD_PREPARED_INPUT_CACHE", "")
    if cache_root:
        from federatedscope.llm.dataset.prepared_input_cache import (
            input_fingerprint, load_or_build)
        fingerprint = input_fingerprint(data_root, tokenizer, config, choice_format)
        return load_or_build(
            cache_root, fingerprint,
            lambda: _load_prepared_shp_table_vii_uncached(
                data_root, tokenizer, config, choice_format))
    return _load_prepared_shp_table_vii_uncached(
        data_root, tokenizer, config, choice_format)


def _load_prepared_shp_table_vii_uncached(data_root, tokenizer, config,
                                         choice_format=True):
    """Load the immutable, checksummed SHP Table VII selector panel."""
    filenames = {
        'train': 'shp_train.jsonl',
        'val': 'shp_validation.jsonl',
        'test': 'shp_test.jsonl',
    }
    paths = {split: os.path.join(data_root, filename)
             for split, filename in filenames.items()}
    missing = [path for path in paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            'Missing prepared SHP Table VII files: ' + ', '.join(missing))
    from federatedscope.llm.dataset.shp_preference import (
        read_prepared_shp_jsonl,
        select_prepared_shp_clients,
        validate_complete_shp_pairs,
        align_shp_response_masks,
        validate_shp_response_masks,
    )
    records = {split: read_prepared_shp_jsonl(path)
               for split, path in paths.items()}
    max_num_test = int(getattr(config.data, 'max_num_test', -1))
    if max_num_test > 0:
        for split in ('val', 'test'):
            records[split] = records[split][:max_num_test]
    expected_clients = int(config.federate.client_num)
    pool_train = bool(getattr(config.data, 'pool_prepared_train', False))
    if pool_train:
        logger.info(
            'SHP centralized head pretraining: pooling %d records from %d '
            'source clients; validation/test remain separate.',
            len(records['train']),
            len({item['category'] for item in records['train']}))
    records['train'] = select_prepared_shp_clients(
        records['train'], expected_clients, pool_all_clients=pool_train)
    train_clients = sorted(
        {item['category'] for item in records['train']},
        key=lambda value: int(value.split('_')[-1]))
    expected_names = [f'Client_{index}' for index in range(expected_clients)]
    if train_clients != expected_names:
        raise ValueError(
            f'Expected prepared clients {expected_names[:2]}...'
            f'{expected_names[-2:]}, observed {len(train_clients)} clients')
    eval_categories = sorted(
        ({item['category'] for item in records['val']}
         | {item['category'] for item in records['test']})
        - set(train_clients))
    category_to_index = {
        category: index for index, category in enumerate(
            train_clients + eval_categories)
    }

    from federatedscope.llm.dataloader.reddit_tldr import (
        WorkerChoiceCausalDataset,
    )
    from federatedscope.llm.dataset.tldr_preference import (
        format_worker_choice_records,
    )

    def _make(items, augment=False):
        if getattr(config.data, 'require_complete_pair_inputs', False):
            validate_complete_shp_pairs(items, tokenizer, SHP_PROMPT_DICT['shp'])
        pair_dataset = LLMComparisonDataset(
            items,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp'],
            prompt_no_input=SHP_PROMPT_DICT['shp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice',
            category_to_index=category_to_index,
            preserve_response_boundary=bool(getattr(
                config.data, 'require_complete_pair_inputs', False)))
        if getattr(config.data, 'require_complete_pair_inputs', False):
            align_shp_response_masks(items, pair_dataset, tokenizer)
            validate_shp_response_masks(items, pair_dataset, tokenizer)
        if not choice_format:
            return pair_dataset
        choice_items = format_worker_choice_records(items, augment=augment)
        choice_dataset = LLMDataset(
            choice_items,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp_cmp'],
            prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
            output_tag='choice')
        choice_dataset.categories = [
            category_to_index[item['category']] for item in choice_items
        ]
        choice_dataset.category_to_index = dict(category_to_index)
        return WorkerChoiceCausalDataset(choice_dataset, pair_dataset)

    datasets_out = (
        _make(records['train'], augment=choice_format),
        _make(records['val']),
        _make(records['test']),
    )
    datasets_out[1].replicate_to_clients = True
    return datasets_out


def load_shp_best_dataset(data_root, tokenizer, config, max_num_test=-1):
    train_dataset, val_dataset, test_dataset = \
        load_comparison_dataset(data_root, tokenizer, config, max_num_test)
    # Use the win_dataset only
    dataset = (train_dataset.dataset_A, val_dataset.dataset_A,
               test_dataset.dataset_A)
    return dataset


def load_shp_cmp_dataset_by_choice(data_root,
                                   tokenizer,
                                   config,
                                   max_num_test=-1):
    token_name = os.path.basename(tokenizer.name_or_path)
    num_clients = config.federate.client_num
    train_fp, val_fp, test_fp = [
        os.path.join(data_root,
                     f'{token_name}_train_choice_{num_clients}.pickle'),
        os.path.join(data_root, f'{token_name}_val_choice.pickle'),
        os.path.join(data_root, f'{token_name}_test_choice.pickle')
    ]

    if os.path.exists(train_fp) and os.path.exists(val_fp) and os.path.exists(
            test_fp):
        with open(train_fp, 'rb') as f_train, open(val_fp, 'rb') as f_val, \
                open(test_fp, 'rb') as f_test:
            train_dataset = pickle.load(f_train)
            val_dataset = pickle.load(f_val)
            test_dataset = pickle.load(f_test)

    else:
        list_train_dict, list_val_dict, list_test_dict = \
            shp_dataset(data_root, num_clients, tokenizer)

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

        train_dataset = LLMDataset(list_train_dict,
                                   tokenizer,
                                   prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                   prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                   output_tag='choice')
        val_dataset = LLMDataset(list_val_dict,
                                 tokenizer,
                                 prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                 prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                 output_tag='choice')
        test_dataset = LLMDataset(list_test_dict,
                                  tokenizer,
                                  prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                  prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                  output_tag='choice')

        # Store these three lists to a pickle file
        with open(train_fp, 'wb') as f_train, \
                open(val_fp, 'wb') as f_val, \
                open(test_fp, 'wb') as f_test:
            pickle.dump(train_dataset, f_train)
            pickle.dump(val_dataset, f_val)
            pickle.dump(test_dataset, f_test)

    # shrink val and test dataset
    if max_num_test > 0:
        val_dataset.input_ids = val_dataset.input_ids[:max_num_test]
        test_dataset.input_ids = test_dataset.input_ids[:max_num_test]

    dataset = (train_dataset, val_dataset, test_dataset)

    return dataset


def load_alpacafarm_human_for_eval(data_root, tokenizer):
    token_name = os.path.basename(tokenizer.name_or_path)
    path = os.path.join(data_root,
                        f'{token_name}_alpacafarm_human_choice.pickle')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            test_dataset = pickle.load(f)
    else:
        ds = datasets.load_dataset("tatsu-lab/alpaca_farm",
                                   "alpaca_human_preference")["preference"]
        list_data_dict = []
        for row in ds.iter(batch_size=1):
            record = {
                "instruction": row["instruction"][0],
                "output_A": row["output_1"][0],
                "output_B": row["output_2"][0],
                "choice": {
                    1: 'A',
                    2: 'B'
                }[row["preference"][0]],
            }
            if row["input"][0]:
                record["instruction"] += f'\n\n{row["input"][0]}'
            list_data_dict.append(record)

        test_dataset = LLMDataset(list_data_dict,
                                  tokenizer,
                                  prompt_input=SHP_PROMPT_DICT['shp_cmp'],
                                  prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
                                  output_tag='choice')

        with open(path, 'wb') as f:
            pickle.dump(test_dataset, f)


# ── FedCQD: comparison dataset with precomputed bias labels ─────────────────

def load_comparison_dataset_with_bias(data_root, tokenizer, config,
                                      max_num_test=-1):
    """
    Loads the bias-enriched SHP comparison files
    (shp_cmpr_{train,val,test}_bias.jsonl) produced by precompute_bias_labels.py.

    Returns LLMComparisonDataset objects that carry .bias_labels,
    which LLMRewardCollator will pass as bias_* keys in every batch.
    """
    token_name  = os.path.basename(tokenizer.name_or_path)
    num_clients = config.federate.client_num

    # Pickled cache to avoid re-tokenisation on repeated runs
    train_fp = os.path.join(data_root,
                            f'{token_name}_train_bias_{num_clients}.pickle')
    val_fp   = os.path.join(data_root, f'{token_name}_val_bias.pickle')
    test_fp  = os.path.join(data_root, f'{token_name}_test_bias.pickle')

    if (os.path.exists(train_fp) and os.path.exists(val_fp)
            and os.path.exists(test_fp)):
        with open(train_fp, 'rb') as ft, \
             open(val_fp,   'rb') as fv, \
             open(test_fp,  'rb') as fx:
            train_dataset = pickle.load(ft)
            val_dataset   = pickle.load(fv)
            test_dataset  = pickle.load(fx)
    else:
        # ── raw bias files ────────────────────────────────────────────────
        bias_kwargs = {
            'instruction': 'instruction',
            'output_A'   : 'output_A',
            'output_B'   : 'output_B',
            'choice'     : 'choice',
            'category'   : 'category',
            'bias_length'          : 'bias_length',
            'bias_first_person'    : 'bias_first_person',
            'bias_action_advice'   : 'bias_action_advice',
            'bias_punct'           : 'bias_punct',
            'bias_negative_framing': 'bias_negative_framing',
            'bias_sentiment'       : 'bias_sentiment',
        }
        train_raw_fp = os.path.join(data_root, 'shp_cmpr_train_bias_100.jsonl')
        val_raw_fp   = os.path.join(data_root, 'shp_cmpr_val_bias.jsonl')
        test_raw_fp  = os.path.join(data_root, 'shp_cmpr_test_bias.jsonl')

        # Fall back to non-bias files if bias files don't exist yet
        if not os.path.exists(train_raw_fp):
            import warnings
            warnings.warn(
                "Bias-labelled SHP files not found. "
                "Run precompute_bias_labels.py first. "
                "Falling back to plain comparison files.")
            return load_comparison_dataset(data_root, tokenizer, config,
                                           max_num_test)

        list_train, list_val, list_test = [], [], []
        for fp, lst in [(train_raw_fp, list_train),
                        (val_raw_fp,   list_val),
                        (test_raw_fp,  list_test)]:
            list_data = load_jsonl(fp, **bias_kwargs)
            lst.extend(list_data)

        # ── LDA client split (same as load_comparison_dataset) ────────────
        list_train_instr, _, _ = _download_shp(data_root)
        cat_idx_map = {}
        for s in list_train_instr:
            if s['category'] not in cat_idx_map:
                cat_idx_map[s['category']] = len(cat_idx_map)
            s['categories'] = cat_idx_map[s['category']]

        splitter = LDASplitter(num_clients, alpha=0.3)
        inst_split = splitter(list_train_instr)
        inst_client = {}
        for idx, sublist in enumerate(inst_split):
            for s in sublist:
                inst_client[s['instruction']] = idx

        for s in list_train:
            s['domain']    = s['category']
            s['category']  = f"Client_{inst_client.get(s['instruction'], 0)}"

        # Filter by token length
        new_train = []
        for s in list_train:
            li = len(tokenizer(s['instruction'])['input_ids'])
            la = len(tokenizer(s['output_A'])['input_ids'])
            lb = len(tokenizer(s['output_B'])['input_ids'])
            if li + la + lb <= config.llm.tok_len:
                new_train.append(s)
        list_train = new_train

        # ── tokenise ─────────────────────────────────────────────────────
        def _make(lst):
            return LLMComparisonDataset(
                lst, tokenizer,
                prompt_input=SHP_PROMPT_DICT['shp'],
                prompt_no_input=SHP_PROMPT_DICT['shp'],
                output_A='output_A', output_B='output_B', choice='choice')

        train_dataset = _make(list_train)
        val_dataset   = _make(list_val)
        test_dataset  = _make(list_test)

        with open(train_fp, 'wb') as ft, \
             open(val_fp,   'wb') as fv, \
             open(test_fp,  'wb') as fx:
            pickle.dump(train_dataset, ft)
            pickle.dump(val_dataset,   fv)
            pickle.dump(test_dataset,  fx)

    # Trim val/test
    if max_num_test > 0:
        for ds in (val_dataset, test_dataset):
            ds.dataset_A.input_ids  = ds.dataset_A.input_ids[:max_num_test]
            ds.dataset_B.input_ids = ds.dataset_B.input_ids[:max_num_test]
            if ds.bias_labels is not None:
                ds.bias_labels = ds.bias_labels[:max_num_test]


def load_llm_labeled_shp(json_path, tokenizer, config=None, max_train=-1):
    """
    Load the LLM-API-labeled SHP dataset (10K samples)
    and convert to LLMComparisonDataset with bias_* fields.

    标签语义：labels 描述 output_A vs output_B 的比较关系。
    - length=yes → output_A 比 output_B 长 25% 以上
    - first_person=yes → output_A 比 output_B 有更多第一人称代词
    - 等等

    构造 SWAP 增广时：
    - d1: A = preferred, B = rejected, choice = 0
      → 若原始 choice=0, A 本就是 preferred, 标签方向一致
      → 若原始 choice=1, preferred=B, 需翻转标签 (1-val)
    - d2: A = rejected, B = preferred, choice = 1
      → bias_labels 同 d1（LLMComparisonDataset swap 后一致）

    改进（vs 前人版本）：
    - 支持全量 10K 标注数据（前人只用 100 条）
    - 删除硬编码路径，val/test 从标注数据自身划分
    - train/val/test = 80%/10%/10%
    - 根据 original_choice 确定 preferred/rejected（修复 choice=1 时方向反的 bug）
    - 标签 = output_A vs output_B → choice=1 时翻转为 preferred vs rejected
    """

    FIELD_MAP = {
        'first_person':  'bias_first_person',
        'length':        'bias_length',
        'action_advice': 'bias_action_advice',
        'punct':         'bias_punct',
        'neg_framing':   'bias_negative_framing',
    }

    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if max_train > 0 and len(raw) > max_train:
        raw = raw[:max_train]

    records = []
    for item in raw:
        base = dict(item['data'])
        lbl = item['labels']

        # 取出原始文本
        raw_a = base.get('output_A')
        raw_b = base.get('output_B')

        # 如果原始字段缺失，直接跳过，防止脏数据报错
        if raw_a is None or raw_b is None:
            continue

        # 根据 original_choice 确定 preferred/rejected
        original_choice = base.get('choice', 0)
        if int(original_choice) == 0:
            preferred, rejected = raw_a, raw_b
        else:
            preferred, rejected = raw_b, raw_a

        # 准备 bias 标签
        # 标签描述 output_A vs output_B
        # d1 的 output_A = preferred，所以需要翻转为 preferred vs rejected
        bias_dict = {}
        for src, dst in FIELD_MAP.items():
            val = 1.0 if str(lbl.get(src, 'no')).lower() == 'yes' else 0.0
            # choice=1 时 preferred=B，标签描述 A vs B = rejected vs preferred → 翻转
            if int(original_choice) == 1:
                val = 1.0 - val
            bias_dict[dst] = val

        # category 兜底
        category = base.get('category', 'unknown')

        # 不把旧 choice 带进去，避免混淆
        base.pop('choice', None)

        # 样本1：A 更好 (A = preferred)
        d1 = dict(base)
        d1['output_A'] = preferred
        d1['output_B'] = rejected
        d1['choice'] = 0
        d1['category'] = category
        d1.update(bias_dict)
        records.append(d1)

        # 样本2：B 更好（交换位置，A = rejected, B = preferred）
        d2 = dict(base)
        d2['output_A'] = rejected
        d2['output_B'] = preferred
        d2['choice'] = 1
        d2['category'] = category
        d2.update(bias_dict)
        records.append(d2)

    # 打乱后按 80/10/10 划分 train/val/test
    random.seed(42)
    random.shuffle(records)
    n = len(records)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_r = records[:n_train]
    val_r = records[n_train:n_train + n_val]
    test_r = records[n_train + n_val:]

    logger.info(
        f"[FedCQD-SHP] Loaded {len(raw)} raw items -> {len(records)} "
        f"augmented records (train={len(train_r)}, val={len(val_r)}, "
        f"test={len(test_r)})"
    )

    def _make(lst):
        return LLMComparisonDataset(
            lst,
            tokenizer,
            prompt_input=SHP_PROMPT_DICT['shp_cmp'],
            prompt_no_input=SHP_PROMPT_DICT['shp_cmp'],
            output_A='output_A',
            output_B='output_B',
            choice='choice'
        )

    return _make(train_r), _make(val_r), _make(test_r)
