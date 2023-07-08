#!/usr/bin/env bash

declare -A model_type_dict
model_type_dict["codegen"]=hf_codegen
model_type_dict["codebert"]=hf_roberta
model_type_dict["graphcodebert"]=hf_roberta
model_type_dict["unixcoder"]=hf_roberta

declare -A pretrained_model_dict
pretrained_model_dict["codegen"]="Salesforce/codegen-350M-mono"
pretrained_model_dict["codebert"]="microsoft/codebert-base"
pretrained_model_dict["graphcodebert"]="microsoft/graphcodebert-base"
pretrained_model_dict["unixcoder"]="microsoft/unixcoder-base"

GPU=${1:-0}
MODEL=${2:-"codegen"}
MODEL_FILE=${3:-"none"}

function lang_eval() {
    if [[ $MODEL_FILE == "none" ]]; then
        result_file_prefix=${MODEL}
    else
        result_dir="$(dirname "${MODEL_FILE}")"
        result_file_prefix=${result_dir}/result
    fi
    for source_lang in ruby python java; do
        for target_lang in ruby python java; do
            result_file=${result_file_prefix}_${source_lang}_${target_lang}.jsonl
            python run.py \
                --model_type ${model_type_dict[$MODEL]} \
                --model_name_or_path ${pretrained_model_dict[$MODEL]} \
                --encoder_state_dict $MODEL_FILE \
                --query_data_file dataset/${source_lang}_with_func.jsonl \
                --candidate_data_file dataset/${target_lang}_with_func.jsonl \
                --result_file $result_file \
                --query_lang ${source_lang} \
                --candidate_lang ${target_lang} \
                --code_length 512 \
                --eval_batch_size 64
        done
    done
}

export CUDA_VISIBLE_DEVICES=$GPU
lang_eval
