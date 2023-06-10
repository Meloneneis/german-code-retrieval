## Tokenizer Adaptation

The `adapt_tokenizer.py` script is used to adapt an existing tokenizer on a new domain. The adaptation process is done via adding new tokens to the tokenizer to better handle the domain by requiring fewer splits for words in the new domain.

### Dependencies

    pip install torch
    pip install transformers

### Example Usage

	python adapt_tokenizer.py \
	    --source_tokenizer microsoft/graphcodebert-base \
	    --target_tokenizer benjamin/roberta-base-wechsel-german \
	    --n_new_tokens 5000 \
	    --output_dir "output_dir"

In this example script, the source tokenizer, which was trained solely on an English corpus, gets adapted to better handle German words by providing a German target tokenizer. 

### Adaptation Process
1.  The script combines the vocabularies of both the source and target tokenizers. If a token exists in both tokenizers, the token and its word embedding from the source tokenizer are prioritized. The indices of the source tokens are retained, resulting in the target tokens having higher indices. This is based on the assumption that the source tokenizer and model were trained more extensively and provide better quality.
    
2.  To create a native merge file required by the transformer's BPE tokenizers, the script uses the tokens from the combined vocabulary as the corpus. The corpus is truncated to match the size of the source vocabulary and the number of newly added tokens. This ensures that the tokens from the source vocabulary are prioritized during the adaptation process.
    
3.  The script trains a new tokenizer using the `train_new_from_iterator()` method provided by transformers. This method creates a new tokenizer and generates a merge file natively. Additional tokens are added to the newly created tokenizer to ensure that all tokens in the new vocabulary can be properly merged.
    
4.  The combined vocabulary is updated to include all tokens from the new tokenizer, enabling the use of the merge file to create the final adapted tokenizer.

### Note
- During the adaptation process, a small amount of tokens is discarded (~200 tokens) due to not being able to convert them to an utf-8 character. This could potentially lead to performance loss.
- It is recommended to use the [WECHSEL library](https://github.com/CPJKU/wechsel) to create the target tokenizer, to ensure that both source and target word embeddings are in the same attention space. Otherwise, the added target tokens are randomly initialized or in a different attention space leading to potentially worse performance.
