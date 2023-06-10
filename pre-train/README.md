## Interleave Datasets for Pre-Training

This script combines multiple datasets into an interleaved dataset file for a pre-training task. The merging process includes preprocessing steps and interleave operations to create an interleaved dataset.

### Dependencies

    pip install tqdm
    pip install datasets
    
### Example Usage

    python interleave_datasets.py \
       --datasets "oscar,wikipedia,code_search_net" \
       --configs "unshuffled_deduplicated_de,20220301.de,all" \
       --distribution "0.2,0.3,0.5" \
       --validation_split_percentage 5 \
       --cache_dir "cache" \
       --output_dir "output_dir"
The parameters `--datasets`, `--configs`, and `--distribution` are mapped by their indices, i.e. the dataset `oscar`, has `unshuffled_deduplicated_de` as config and `0.2` as distribution.
		    

### Note:
- Using the scripts on datasets other than wikipedia, oscar or code_search_net might not work!
- Only RoBERTa tokenizers are supported when merging `code_search_net` dataset as the preprocessing consists of linking the documentation and code via the RoBERTa separator token

