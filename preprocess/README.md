# Data

### Obtaining the Raw Data

* For programming language data: [ TBD ]

* For natural language data, we use WikiText-103 for pretraining. This data can be obtained by following the instructions [here](https://github.com/yxuansu/SimCTG/tree/main/data#1-download-wikitext-103-dataset) [1].

### Preprocessing

This folder contains scripts to preprocess the PL and NL datasets. Assuming access to raw text data, our multi-stage preprocessing follows these steps:


#### 1. Generating Splits

Each row in the dataset is shuffled and split into train/valid/test spilts. The ratios of the splits can be customized. Each split is stored as a `.arrow` dataset using huggingface `datasets` library.


#### 2. Chunking & Tokenization

Each raw string in each split from above is chunked up to a custom threshold. For PL data, we chunk up to the last line (i.e., finding `\n`) such that the resulting chunk has <= `max_chars_per_token * sequence_length` characters. This chunk is tokenized and the resulting dataset is stored as a `.arrow` dataset using huggingface `datasets` library.

#### 3. Exact Match (EM) Deduplication

After chunking, the resulting datasets may have duplicate strings. To eliminate these, we simply drop duplicate strings based on EM. The summary of this step is `output_data_strs := set(input_data_strs)`.

## Usage

* For programming language data, set the relevant paths and variables in `run_preprocess_bq.sh` and run:
```shell
bash run_preprocess_bq.sh
```

* For natural language data, set the relevant paths and variables in `run_preprocess_wikitext.sh` and run:
```shell
bash run_preprocess_wikitext.sh
```

#### References

[1] Su, Yixuan, et al. "A contrastive framework for neural text generation." arXiv preprint arXiv:2202.06417 (2022).
