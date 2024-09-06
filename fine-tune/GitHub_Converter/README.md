## Convert GitHub Repositories to a dataset

This script converts Github repositories to a func-and-documentation jsonl dataset. This script leverages the function_parser library to create the dataset.

### Dependencies

    pip install -r requirements.txt
    
### Example Usage

    python convert_repos_to_docs_and_funcs.py \
       --lang "java" \
       --path_to_repos "../../data/github/german_repository_names.json" \
       --output_dir "../../data/github/raw_german_docs_and_funcs.jsonl" \
       --spoken_language "de"



### Note
 - Some columns need to be renamed, so that it can be used with the fine-tuning script [run.py](https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/run.py).