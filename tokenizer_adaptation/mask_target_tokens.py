import tqdm

from transformers import AutoTokenizer, AutoModelForMaskedLM, RobertaForMaskedLM, RobertaTokenizerFast, Trainer
import argparse
import pickle
from datasets import load_dataset
import random


def get_source_ratio(encoded_text, source_mask):
    source_mask_set = set(source_mask)
    total_count_encoded_text = len(encoded_text)
    count_source_idx_in_encoded_text = sum(1 for idx in encoded_text if idx in source_mask_set)
    ratio = count_source_idx_in_encoded_text / total_count_encoded_text if total_count_encoded_text else 0
    return ratio

def flatten_list(lst):
    """
    This function takes a list which may contain sublists and returns a new list with the sublists' elements
    merged into the parent list, effectively removing the nested list structure.
    """
    flat_list = []
    for item in lst:
        if isinstance(item, list):  # Check if the item is a list
            flat_list.extend(item)  # If so, extend the flat_list by adding elements of the sublist
        else:
            flat_list.append(item)  # If not, just append the item itself
    return flat_list

def main():
    parser = argparse.ArgumentParser(description="Mask target tokens")

    parser.add_argument("--tokenizer", type=str, required=False, default="adapted_model_and_tok")
    parser.add_argument("--source_tokenizer", type=str, default="FacebookAI/roberta-base")
    parser.add_argument("--mask_path", type=str, required=False, default="adapted_model_and_tok/source_indices.pkl")
    parser.add_argument("--dataset_name", type=str, default="mlsum")
    parser.add_argument("--dataset_config", type=str, default="de")
    # parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--source_token_ratio", type=float, default=0.95)
    args = parser.parse_args()

    with open(args.mask_path, 'rb') as handle:
        source_indices = pickle.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    source_tokenizer = AutoTokenizer.from_pretrained(args.source_tokenizer)
    dataset = load_dataset(args.dataset_name, args.dataset_config, cache_dir="cache")
    column_names = dataset["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]
    source_indices_set = set(source_indices)

    def tokenize_function(examples):
        encoded_text = tokenizer(examples["text"], return_special_tokens_mask=True)
        source_ratio = get_source_ratio(encoded_text["input_ids"], source_indices)
        if source_ratio < args.source_token_ratio:
            # mask out random new tokens with old tokens til source_ratio >= args.source_token_ratio
            # Calculate the number of tokens to mask to reach the target ratio
            num_tokens_to_mask = int(len(encoded_text["input_ids"]) * (args.source_token_ratio - source_ratio))
            target_tokens = [(pos, idx) for pos, idx in enumerate(encoded_text["input_ids"]) if idx not in source_indices_set]
            random.shuffle(target_tokens)
            mask_target_tokens = target_tokens[:num_tokens_to_mask]
            for i, (pos, token) in enumerate(mask_target_tokens):
                decoded_mask_target_token = tokenizer.decode(token)
                encoded_new_source_tokens = source_tokenizer.encode(decoded_mask_target_token, add_special_tokens=False)
                encoded_text["input_ids"][pos] = encoded_new_source_tokens
                attention_array = [1 for _ in range(len(encoded_new_source_tokens))]
                special_token_array = [0 for _ in range(len(encoded_new_source_tokens))]
                encoded_text["attention_mask"][pos] = attention_array
                encoded_text["special_tokens_mask"][pos] = special_token_array
            encoded_text["input_ids"] = flatten_list(encoded_text["input_ids"])
            encoded_text["attention_mask"] = flatten_list(encoded_text["attention_mask"])
            encoded_text["special_tokens_mask"] = flatten_list(encoded_text["special_tokens_mask"])
        return encoded_text

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=False,
        remove_columns=column_names,
        desc="Tokenizing dataset"
    )


if __name__ == "__main__":
    main()
