import json
from transformers import AutoTokenizer, AutoModel, RobertaForMaskedLM, RobertaTokenizerFast
import argparse
import torch
import collections


def main():
    parser = argparse.ArgumentParser(description="Adapt an existing tokenizer")

    parser.add_argument("--source_tokenizer", type=str, required=True)
    parser.add_argument("--target_tokenizer", type=str, required=True)
    parser.add_argument("--n_new_tokens", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default="")
    args = parser.parse_args()

    target_tok = AutoTokenizer.from_pretrained(args.target_tokenizer)
    target_model = AutoModel.from_pretrained(args.target_tokenizer)

    source_tok = AutoTokenizer.from_pretrained(args.source_tokenizer)
    source_model = AutoModel.from_pretrained(args.source_tokenizer)

    # create dict with token index, token string, language, and embedding for english and german
    target_vocab = {k: {"index": v, "string": target_tok.convert_tokens_to_string([k]) if target_tok.convert_tokens_to_string([k]) != "�" else k, "model": "target_lang",
                        "embedding": target_model.embeddings.word_embeddings.weight[v]} for k, v in
                    target_tok.get_vocab().items()}
    target_vocab = {k: v for k, v in sorted(target_vocab.items(), key=lambda item: item[1]["index"])}
    source_vocab = {k: {"index": v, "string": source_tok.convert_tokens_to_string([k]), "model": "source_lang",
                        "embedding": source_model.embeddings.word_embeddings.weight[v]} for k, v in
                    source_tok.get_vocab().items()}
    source_vocab = {k: v for k, v in sorted(source_vocab.items(), key=lambda item: item[1]["index"])}

    # combine german and english vocab to get a all the existing token embeddings in one dict
    # this is the final vocab to be used for the final tokenizer and final model
    target_vocab.update(source_vocab)
    final_vocab = {k: v for k, v in sorted(target_vocab.items(), key=lambda item: item[1]["model"], reverse=True)}

    # for some tokens we cannot decode utf-8 chars -> delete them
    final_vocab = {k: v for k, v in sorted(final_vocab.items(), key=lambda item: item[1]["index"]) if "�" not in v["string"]}

    # reorder indíces, prioritize retaining indices of source tokenizer
    all_numbers = set(i for i in range(len(final_vocab)))
    indexes = [value["index"] for value in final_vocab.values()]
    missing_idx = all_numbers.difference(set(indexes))
    duplicate_idx = [item for item, count in collections.Counter(indexes).items() if count > 1]
    zipped = dict(zip(duplicate_idx, missing_idx))
    temp_vocab = {
        k: {"index": zipped[v["index"]], "string": v["string"], "model": "target_lang", "embedding": v["embedding"]}
        for k, v in final_vocab.items() if v["index"] in zipped.keys() and v["model"] == "target_lang"}
    final_vocab.update(temp_vocab)
    final_vocab = {k: v for k, v in sorted(final_vocab.items(), key=lambda item: item[1]["index"])}

    corpus = [v["string"] for k, v in final_vocab.items()][:(source_tok.vocab_size + args.n_new_tokens + 10)]  # +10 because of special tokens
    # creating a tokenizer ONLY for the merge file as we can create it natively
    tokenizer = source_tok.train_new_from_iterator(corpus, source_tok.vocab_size + args.n_new_tokens)
    tokenizer.save_pretrained("tokenizer")

    tok_vocab = tokenizer.get_vocab()
    tok_vocab = {k: v for k, v in sorted(tok_vocab.items(), key=lambda item: item[1])}
    tok_vocab_set = set(k for k, v in tok_vocab.items())
    combined_vocab_set = set(k for k, v in final_vocab.items())
    difference = tok_vocab_set.difference(combined_vocab_set)  # for these tokens, embeddings dont exist
    union = tok_vocab_set - difference  # for these tokens, embeddings exist
    difference2 = set(k for k, v in
                      final_vocab.items()) - union  # these are the tokens embeddings are included in vocab, but are not needed

    # delete all the unneeded tokens with embeddings
    for key in difference2:
        final_vocab.pop(key)

    def get_embedding(token):
        word = tokenizer.convert_tokens_to_string([token])
        encodings = source_tok.encode(word, add_special_tokens=False)
        embeddings = [source_model.embeddings.word_embeddings.weight[encoding] for encoding in encodings]
        with torch.no_grad():
            embedding = sum(embeddings) / len(embeddings)
        return embedding

    all_numbers = set(i for i in range(source_tok.vocab_size + args.n_new_tokens))
    indexes = [value["index"] for value in final_vocab.values()]
    missing_idx = sorted(list(all_numbers.difference(set(indexes))))
    subset_missing_idx = missing_idx[:len(difference)]
    zipped = zip(difference, subset_missing_idx)
    temp = {pair[0]: {
        "index": pair[1],
        "string": None,
        "model": None,
        "embedding": get_embedding(pair[0])
    } for pair in zipped}
    final_vocab.update(temp)

    indexes = [value["index"] for value in final_vocab.values()]
    missing_idx = sorted(list(all_numbers.difference(set(indexes))))
    tokens_with_overflow_idx = {k: v for k, v in final_vocab.items() if v["index"] >= len(all_numbers)}
    for token in tokens_with_overflow_idx:
        final_vocab[token]["index"] = missing_idx.pop()

    final_model = RobertaForMaskedLM.from_pretrained(args.source_tokenizer)
    final_model.resize_token_embeddings(len(tokenizer))
    for token in union:
        if final_vocab[token]["model"] != "source_lang":
            with torch.no_grad():
                final_model.roberta.embeddings.word_embeddings.weight[final_vocab[token]["index"]] = final_vocab[token]["embedding"]

    final_vocab = {k: v["index"] for k, v in final_vocab.items()}
    with open("vocab.json", "w") as fp:
        json.dump(final_vocab, fp)
    final_tok = RobertaTokenizerFast(vocab_file="vocab.json", merges_file="tokenizer/merges.txt",
                                     model_max_length=tokenizer.model_max_length)
    final_tok.save_pretrained(args.output_dir)
    final_model.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
