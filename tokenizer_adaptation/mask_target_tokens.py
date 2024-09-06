from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, RobertaTokenizerFast, Trainer
import pickle
from datasets import load_dataset, load_from_disk
import random

# HYPERPARAMETERS
#####################################################################################
SOURCE_TOKEN_RATIO = 1.0
SOURCE_TOKENIZER_CHECKPOINT = "microsoft/graphcodebert-base"
TARGET_TOKENIZER_CHECKPOINT = "adapted_model_and_tok"
LOAD_FROM_DISK = True
DATASET_CHECKPOINT = "UONLPEnglish"
DATASET_CONFIG = "de-en"
OUTPUT_DIR = "UONLPEnglishSTR100"
TEXT_COLUMN_NAME = "text"
######################################################################################

source_tokenizer = AutoTokenizer.from_pretrained(SOURCE_TOKENIZER_CHECKPOINT)
target_tokenizer = AutoTokenizer.from_pretrained(TARGET_TOKENIZER_CHECKPOINT)
if LOAD_FROM_DISK:
    raw_datasets = load_from_disk(DATASET_CHECKPOINT)
else:
    raw_datasets = load_dataset(DATASET_CHECKPOINT, DATASET_CONFIG)

if "test" in raw_datasets:
    del raw_datasets["test"]

with open(f'{TARGET_TOKENIZER_CHECKPOINT}/source_indices.pkl', 'rb') as f:
    source_indices = pickle.load(f)
    source_indices_set = set(source_indices)


def process(example):
    max_length = 512
    if len(example["input_ids"]) > max_length:
        example["input_ids"] = example["input_ids"][:max_length - 1]  # Truncate input_ids
        example["attention_mask"] = example["attention_mask"][:max_length]  # Truncate attention_mask
        example["input_ids"].append(2)  # Append end of sequence token
    elif len(example["input_ids"]) < max_length:
        padding_length = max_length - len(example["input_ids"])  # Calculate padding length
        example["input_ids"] += [1] * padding_length  # Pad input_ids with zeros
        example["attention_mask"] += [0] * padding_length  # Pad attention_mask with zeros
    assert len(example["input_ids"]) == max_length, "Input Ids are not max length!"
    assert len(example["attention_mask"]) == max_length, "Attention mask are not max length!"
    return example


def truncate_list(lst, threshold):
    if len(lst) <= threshold:
        return lst
    else:
        return lst[:threshold - 1] + [lst[-1]]



def get_source_ratio(encoded_text, source_mask):
    source_mask_set = set(source_mask)
    total_count_encoded_text = len(encoded_text)
    count_source_idx_in_encoded_text = sum(1 for idx in encoded_text if idx in source_mask_set)
    ratio = count_source_idx_in_encoded_text / total_count_encoded_text if total_count_encoded_text else 0
    return ratio


def flatten_list(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):  # Check if the item is a list
            flat_list.extend(item)  # If so, extend the flat_list by adding elements of the sublist
        else:
            flat_list.append(item)  # If not, just append the item itself
    return flat_list


source_ratios = []


def tokenize_function(examples):
    encoded_text = target_tokenizer(examples[TEXT_COLUMN_NAME], return_special_tokens_mask=True)
    source_ratio = get_source_ratio(encoded_text["input_ids"], source_indices)
    source_ratios.append(source_ratio)
    if source_ratio < SOURCE_TOKEN_RATIO:
        # mask out random new tokens with old tokens til source_ratio >= args.source_token_ratio
        # Calculate the number of tokens to mask to reach the target ratio
        num_tokens_to_mask = int(len(encoded_text["input_ids"]) * (SOURCE_TOKEN_RATIO - source_ratio))
        target_tokens = [(pos, idx) for pos, idx in enumerate(encoded_text["input_ids"]) if
                         idx not in source_indices_set]
        random.shuffle(target_tokens)
        mask_target_tokens = target_tokens[:num_tokens_to_mask]
        for i, (pos, token) in enumerate(mask_target_tokens):
            decoded_mask_target_token = target_tokenizer.decode(token)
            encoded_new_source_tokens = source_tokenizer.encode(decoded_mask_target_token,
                                                                add_special_tokens=False)
            encoded_text["input_ids"][pos] = encoded_new_source_tokens
            attention_array = [1 for _ in range(len(encoded_new_source_tokens))]
            special_token_array = [0 for _ in range(len(encoded_new_source_tokens))]
            encoded_text["attention_mask"][pos] = attention_array
            encoded_text["special_tokens_mask"][pos] = special_token_array
        encoded_text["input_ids"] = flatten_list(encoded_text["input_ids"])
        encoded_text["attention_mask"] = flatten_list(encoded_text["attention_mask"])
        encoded_text["special_tokens_mask"] = flatten_list(encoded_text["special_tokens_mask"])
    return encoded_text


def tokenize_function_eval(examples):
    return target_tokenizer(examples[TEXT_COLUMN_NAME], truncation=True, padding="max_length",
                            return_special_tokens_mask=True)


if "train" not in raw_datasets:
    raw_datasets = raw_datasets.map(
        tokenize_function if SOURCE_TOKEN_RATIO != 0.0 else tokenize_function_eval,
        batched=False,
        remove_columns=[TEXT_COLUMN_NAME],
        desc="Running tokenizer on dataset line_by_line",
        load_from_cache_file=False,
        num_proc=64
    )
else:
    raw_datasets["train"] = raw_datasets["train"].map(
        tokenize_function if SOURCE_TOKEN_RATIO != 0.0 else tokenize_function_eval,
        batched=False,
        remove_columns=[TEXT_COLUMN_NAME],
        desc="Running tokenizer on dataset line_by_line",
        load_from_cache_file=False,
        num_proc=64
    )
if "validation" in raw_datasets:
    raw_datasets["validation"] = raw_datasets["validation"].map(
        tokenize_function_eval,
        batched=False,
        remove_columns=[TEXT_COLUMN_NAME],
        desc="Running tokenizer on dataset line_by_line",
        load_from_cache_file=False,
        num_proc=64
    )

raw_datasets = raw_datasets.map(process, load_from_cache_file=False, num_proc=64)
raw_datasets.save_to_disk(OUTPUT_DIR)
