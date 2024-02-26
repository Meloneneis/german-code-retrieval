from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, RobertaTokenizerFast, Trainer
import argparse
from tqdm import tqdm
import pickle
from datasets import load_dataset
import random
import torch
import json
from itertools import chain


def get_source_ratio(encoded_text, source_mask):
    total_count_encoded_text = len(encoded_text)
    count_source_idx_in_encoded_text = sum(1 for idx in encoded_text if idx in source_mask)
    if total_count_encoded_text == 0:
        ratio = 0
    else:
        ratio = count_source_idx_in_encoded_text / total_count_encoded_text
    return ratio


def main():
    parser = argparse.ArgumentParser(description="Mask target tokens")

    parser.add_argument("--tokenizer", type=str, required=False, default="adapted_model_and_tok")
    parser.add_argument("--source_tokenizer", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--mask_path", type=str, required=False, default="adapted_model_and_tok/source_indices.pkl")
    parser.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--dataset_config", type=str, default="20231101.de")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--source_token_ratio", type=float, default=0.95)
    args = parser.parse_args()

    with open(args.mask_path, 'rb') as handle:
        source_indices = pickle.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.dataset_split, cache_dir="cache")
    dataset = dataset.select(range(100000))
    max_seq_length = tokenizer.model_max_length
    column_names = dataset.column_names

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of
    # max_seq_length.
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    def tokenize_function(examples):
        encoded_text = tokenizer(examples["text"], return_special_tokens_mask=True)
        source_ratio = get_source_ratio(encoded_text["input_ids"], source_indices)
        if source_ratio < args.source_token_ratio:
            # mask out random new tokens with old tokens til source_ratio >= args.source_token_ratio
            # Calculate the number of tokens to mask to reach the target ratio
            num_tokens_to_mask = int(len(encoded_text) * (args.source_token_ratio - source_ratio))
            target_tokens = [idx for idx in encoded_text if idx not in source_indices]
            random.shuffle(target_tokens)
            mask_target_tokens = target_tokens[:num_tokens_to_mask]
            decoded_mask_target_tokens = tokenizer.decode(mask_target_tokens)
            encoded_new_source_tokens = source_tokenizer.encode(decoded_mask_target_tokens, add_special_tokens=False)
            print("S")
        return encoded_text

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Tokenizing dataset"
    )
    tokenized_datasets = iter(tokenized_datasets)
    for _ in tqdm(range(10000), desc="Tokenizing dataset"):
        dataset.append(next(tokenized_datasets))
    print("Hellp")

'''    def save_as_json_lines(strings, filename):
        with open(filename, 'w') as f:
            for string in strings:
                json.dump({'text': string}, f)
                f.write('\n')
    save_as_json_lines(encoded_text, "encoded_texts.jsonl")'''


if __name__ == "__main__":
    main()
