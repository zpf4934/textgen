# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import pickle
import re

import datasets
import numpy as np
from datasets import Dataset as HFDataset
from datasets import load_dataset
from loguru import logger
from rouge import Rouge
from torch.utils.data import Dataset
from tqdm.auto import tqdm

PROMPT_DICT = {
    "prompt_input": (
        "问：{instruction}\n{input_text}\n答："
    ),
    "prompt_no_input": (
        "问：{instruction}\n答："
    ),
    "prompt_multi_round_no_input": (
        "问：{instruction}{output_text}"
    ),
}


def generate_prompt(instruction, input_text, output_text):
    """Generate prompt for instruction."""
    if 'Human:' in instruction and 'Assistant:' in instruction:
        instruction = instruction.replace('Human:', '### Human: ')
        instruction = instruction.replace('Assistant:', '### Assistant: ')
        prompt = PROMPT_DICT['prompt_multi_round_no_input'].format(instruction=instruction, output_text=output_text)
        return prompt, 'multi_round'
    else:
        if input_text:
            prompt = PROMPT_DICT["prompt_input"].format(instruction=instruction, input_text=input_text)
        else:
            prompt = PROMPT_DICT["prompt_no_input"].format(instruction=instruction)
        return prompt, 'single_round'


def preprocess_data(data):
    instruction, input_text, target_text, tokenizer, args = data
    IGNORE_INDEX = -100
    EOS_TOKEN = tokenizer.eos_token

    prompt, round_type = generate_prompt(instruction, input_text, target_text)
    if round_type == 'multi_round':
        prompt = re.sub(r'(?<!\n)\n### ', f'\n{EOS_TOKEN}### ', prompt)
        prompt += EOS_TOKEN
        example = tokenizer(prompt, return_offsets_mapping=True)
        labels = example['input_ids'].copy()
        if not args.is_train_on_prompt:
            source_len = len(tokenizer(
                PROMPT_DICT['prompt_multi_round_no_input'].split('\n\n')[0] + '\n\n')['input_ids'])
            labels[:source_len] = [IGNORE_INDEX] * source_len
            offsets = example["offset_mapping"]

            matches = re.finditer(r'### (?!Assistant:)(.*?)<\/s>', prompt, re.DOTALL)
            for match in matches:
                start_pos, end_pos = match.span()
                start_idx = None
                end_idx = None

                for i, (start, end) in enumerate(offsets):
                    if start <= start_pos < end:
                        start_idx = i
                    if start <= end_pos < end:
                        end_idx = i

                if start_idx is not None and end_idx is not None:
                    for i in range(start_idx, end_idx - 1):
                        labels[i] = -100

            example['labels'] = labels
        return example
    else:
        full_prompt = prompt + target_text + tokenizer.eos_token
        full_max_length = args.max_seq_length + args.max_length
        example = tokenizer(
            full_prompt,
            truncation=True,
            max_length=full_max_length,
            padding=False,
            add_special_tokens=False
        )
        example["labels"] = example["input_ids"].copy()
        if not args.is_train_on_prompt:
            user_example = tokenizer(
                prompt,
                truncation=True,
                max_length=args.max_seq_length,
                padding=False,
                add_special_tokens=False
            )
            user_prompt_len = len(user_example["input_ids"])
            # set labels to full max length to adjust for DataCollatorForSeq2Seq padding
            example["labels"] = [-100] * (full_max_length - len(example['labels']) + user_prompt_len) + \
                                example["labels"][user_prompt_len:]
        return example


def preprocess_batch_for_hf_dataset(dataset, tokenizer, args):
    data = (dataset["instruction"], dataset["input"], dataset["output"], tokenizer, args)
    dataset = preprocess_data(data)
    return dataset


def load_hf_dataset(data, tokenizer, args, mode):
    if isinstance(data, str):
        if data.endswith('.json') or data.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=data)
        elif os.path.isdir(data):
            dataset = datasets.load_from_disk(data)
        else:
            dataset = load_dataset(
                data,
                download_mode="force_redownload"
                if args.reprocess_input_data
                else "reuse_dataset_if_exists",
            )
        # This is not necessarily a train dataset. The datasets library insists on calling it train.
        dataset = dataset['train']
        if mode == 'dev' and args.max_eval_samples is not None:
            max_eval_samples = min(len(dataset), args.max_eval_samples)
            dataset = dataset.select(range(max_eval_samples))
    else:
        dataset = HFDataset.from_pandas(data)

    dataset = dataset.shuffle().map(
        lambda x: preprocess_batch_for_hf_dataset(x, tokenizer=tokenizer, args=args),
        batched=False, remove_columns=dataset.column_names
    ).filter(lambda x: tokenizer.gmask_token_id in list(x['input_ids']))  # exclude samples without gmask

    return dataset


class ChatGlmDataset(Dataset):
    def __init__(self, tokenizer, args, data, mode):
        cached_features_file = os.path.join(
            args.cache_dir,
            args.model_name.replace("/", "_")
            + "_cached_"
            + str(args.max_seq_length)
            + str(len(data)),
        )

        if os.path.exists(cached_features_file) and (
                (not args.reprocess_input_data and not args.no_cache)
                or (mode == "dev" and args.use_cached_eval_features and not args.no_cache)
        ):
            logger.info(" Loading features from cached file %s" % cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.debug(" Creating features from dataset file at %s" % args.cache_dir)
            data = [
                (instruction, input_text, target_text, tokenizer, args)
                for instruction, input_text, target_text in zip(
                    data["instruction"], data["input"], data["output"]
                )
            ]
            self.examples = [preprocess_data(d) for d in tqdm(data, disable=args.silent)]
            if not args.no_cache:
                logger.info(" Saving features into cached file %s" % cached_features_file)
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def compute_bleu(label, pred, weights=None):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    weights = weights or (0.25, 0.25, 0.25, 0.25)
    return np.mean([sentence_bleu(references=[list(a)], hypothesis=list(b),
                                  smoothing_function=SmoothingFunction().method1, weights=weights)
                    for a, b in zip(label, pred)])


def compute_rouge(label, pred, weights=None, mode='weighted'):
    weights = weights or (0.2, 0.4, 0.4)
    if isinstance(label, str):
        label = [label]
    if isinstance(pred, str):
        pred = [pred]
    label = [' '.join(x) for x in label]
    pred = [' '.join(x) for x in pred]

    def _compute_rouge(label, pred):
        try:
            scores = Rouge().get_scores(hyps=label, refs=pred)[0]
            scores = [scores['rouge-1']['f'], scores['rouge-2']['f'], scores['rouge-l']['f']]
        except ValueError:
            scores = [0, 0, 0]
        return scores

    scores = np.mean([_compute_rouge(*x) for x in zip(label, pred)], axis=0)
    if mode == 'weighted':
        return {'rouge': sum(s * w for s, w in zip(scores, weights))}
    elif mode == '1':
        return {'rouge-1': scores[0]}
    elif mode == '2':
        return {'rouge-2': scores[1]}
    elif mode == 'l':
        return {'rouge-l': scores[2]}
    elif mode == 'all':
        return {'rouge-1': scores[0], 'rouge-2': scores[1], 'rouge-l': scores[2]}
