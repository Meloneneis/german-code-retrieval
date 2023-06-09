Interleave Dataset for Pretraining

This tool combines multiple datasets into an interleaved dataset file for a pretraining task with masked language modeling (MLM). The merging process includes preprocessing steps and interleave operations to create an interleaved dataset.

Functionality:
- Loads and preprocesses datasets
- Interleaves the datasets
- Saves the interleaved dataset to a file

Usage:
1. Prepare the datasets:
   - The tool supports merging multiple datasets for pretraining.
   - Specify the datasets to merge and their corresponding configurations in the code.
   - Make sure the necessary datasets and configurations are available for loading.

2. Run the tool:
   - Execute the script to start the merging process.
   - The tool will load and preprocess the datasets, including any required preprocessing steps such as adding columns, mapping functions, or filtering examples.
   - The datasets will be interleaved based on a specified distribution.
   - Finally, the interleaved dataset will be saved to the specified output directory.

Note:
- Using the scripts on datasets other than wikipedia, oscar or code_search_net might not work!
- The tool assumes the availability of the necessary datasets and configurations.
- Customize the code and configurations according to your specific datasets and requirements.

