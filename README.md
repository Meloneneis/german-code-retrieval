# German Code Search

This repository contains code and data for the paper Adapting GraphCodeBERT for German Code Retrieval.

## Directories

### Tokenizer Adapatation

The `tokenizer_adaptation` directory contains a script to adapt a source tokenizer and a model to better handle a 
new domain via a target tokenizer and model that was already trained on the new domain.

### Pre-Train

The `Pre-Train` directory contains code for interleaving multiple datasets.

For more detailed information, refer to the [Pretraining README](src/Pretraining/README.md).

To perform the masked language modeling pretraining task, you can refer to the Hugging Face Transformers library's 
official script: 
[run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py).

### Fine-Tune

The `fine-tune` directory contains a `GitHub_Scraper` for collecting GitHub repositories with specific configurations. 
The directory also contains a `GitHub_Converter` to convert GitHub repositories to a doc-and-func dataset for fine-tuning.

To perform the fine-tuning task, you can refer to the Microsoft CodeBERT project's script: [run.py](https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/run.py).

### Data
The `data` directory contains an already scraped and converted German GitHub java dataset that can be used for fine-tuning.

## Workflow

To use the code and data in this repository, follow these steps:
   
1. Adapt the Tokenizer: Refer to the `tokenizer_adaptation` directory for detailed information.

2. Create a Pre-Training Dataset: Refer to the `pre-train` directory for more information.

3. Run the Pre-Training script with the adapted tokenizer and pretraining dataset: [run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py).

4. Scrape a Fine-Tuning Dataset from GitHub: Refer to the `fine-tune` directory for more information.

5. Run the Fine-Tuning script with the further pre-trained model and adapted tokenizer on the scraped GitHub Dataset: [run.py](https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/run.py).


