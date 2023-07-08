# Code-to-Code Search

Given a source code as the query, the task aims to retrieve codes with the same semantics from a collection of candidates in zero-shot setting. We collect 11,744/15,594/23,530 functions from [CodeNet](https://github.com/IBM/Project_CodeNet) corpus in Ruby/Python/Java. Each function solves one of 4,053 problems.

## Data Download

```bash
mkdir dataset
cd dataset
wget https://dax-cdn.cdn.appdomain.cloud/dax-project-codenet/1.0.0/Project_CodeNet.tar.gz
tar -xvf Project_CodeNet.tar.gz
python preprocess.py
cd ..
```

## Evaluation

To run evaluation, follow the template command:
```shell
bash eval.sh <GPU_ID> <MODEL_NAME> <PATH_TO_CKPT_PT_FILE>
```
`<MODEL_NAME>` options are: `codegen, codebert, graphcodebert, unixcoder`.
For more details see `eval.sh`.

`eval.sh` will execute `run.py`. An example of the script call:

```bash
source_lang=ruby
target_lang=python
python run.py \
--model_name_or_path microsoft/unixcoder-base  \
--query_data_file dataset/${source_lang}_with_func.jsonl \
--candidate_data_file dataset/${target_lang}_with_func.jsonl \
--query_lang ${source_lang} \
--candidate_lang ${target_lang} \
--code_length 512 \
--eval_batch_size 256 
```


