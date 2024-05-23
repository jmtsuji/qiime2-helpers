# qiime2-helpers
Repo with helper scripts for QIIME2 analyses

All scripts can be run through the Linux command line.

## Contents

### `generate_combined_feature_table.py`
Adds taxonomy and representative sequence information onto a TSV-format FeatureTable

#### Dependencies
- python >= 3
- pandas
- biopython

#### Installation
First, add the script to your `PATH`. 
Then, install all dependencies. If using conda:
```bash
conda create -n generate_combined_feature_table -c conda-forge \
  python=3 pandas biopython

conda activate generate_combined_feature_table
```

#### Testing
If interested, run the automated end-to-end test with:
```bash
testing/generate_combined_feature_table/test_generate_combined_feature_table.sh \
  testing/generate_combined_feature_table
```

#### Usage
Example usage (test data in repo):
```bash
# Assuming you are in the Github repo directory
input_dir="testing/generate_combined_feature_table/inputs"

generate_combined_feature_table.py \
  -f "${input_dir}/feature_table.tsv" \
  -s "${input_dir}/representative_seqs.fasta" \
  -t "${input_dir}/taxonomy.tsv" \
  -o "test_table.tsv" \
  --parse_taxonomy
```
run `generate_combined_feature_table.py -h` for full usage instructions.

### `summarize_sample_counts.py`
Summarizes total counts per sample for a QZA FeatureCounts[Frequency] archive. 
This script is nice if trying to decide how to rarefy your data, for example.

#### Dependencies
- python >= 3
- pandas >= 2
- biom-format = 2.1
- h5py = 2.9

#### Installation
First, add the script to your `PATH`. 
Then, install all dependencies. If using conda:
```bash
conda create -n summarize_sample_counts -c conda-forge -c bioconda \
  python=3 pandas=2 biom-format=2.1 h5py=2.9

conda activate summarize_sample_counts
```

#### Testing
TODO

#### Usage
Run `summarize_sample_counts.py -h` for full usage instructions.

### Other scripts
Docs coming soon...