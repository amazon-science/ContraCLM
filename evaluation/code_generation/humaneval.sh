ckpt_dir_options=(
    /path/to/ckpt/dir
)
ckpt_path_options=(
    /rel/path/to/ckpt
)

PRETRAINED_EVAL='' # if you want to eval ContraCLM models
# PRETRAINED_EVAL='--eval_pretrained' # if you want to eval pretrained model

temperatures=(0.2)
devices=(0)

RANKED_EVAL='' # turn off ranked eval
# RANKED_EVAL='--ranked_pass_at_k' # ranked eval

NUM_SAMPLES=50
MAX_COMP_LENGTH=300
BATCH_CNT=4
TOK_MAX_LEN=512

for i in 0
do
    CKPT_DIR=${ckpt_dir_options[$i]}
    MODEL_PATH=${ckpt_path_options[$i]}

    echo ========================================================
    echo Evaluating ${CKPT_DIR}
    echo ========================================================

    # generate sequences
    for j in 0
    do
        T=${temperatures[$j]}
        DEV=${devices[$j]}
        DECODING='--decoding_strategy temperature_sampling --sampling_temperature '$T

        python humaneval.py \
            --model_name Salesforce/codegen-350M-mono \
            --pad_token_id 50256 \
            --ckpt_dir $CKPT_DIR \
            --model_path $MODEL_PATH \
            $PRETRAINED_EVAL \
            --seed 1234 \
            --accelerator gpu \
            $DECODING \
            --eval_device $DEV \
            --deepspeed_ckpt \
            $RANKED_EVAL \
            --num_samples_per_task $NUM_SAMPLES \
            --max_completion_length $MAX_COMP_LENGTH \
            --tok_max_length $TOK_MAX_LEN \
            --batch_cnt $BATCH_CNT &
    done
    wait

    # evaluate functional correctness
    for t in 0.2
    do
        if [[ $RANKED_EVAL == "--ranked_pass_at_k" ]]; then
            # pass@k, ranked pass@k evaluation
            python human_eval/evaluate_functional_correctness.py \
                --task humaneval \
                --sample_file $CKPT_DIR/humaneval_samples_temperature_sampling_${t}_ranked_eval_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.json \
                --score_field mean_logp \
                --k '1,5,10,100' \
                --exec_check=False \
                > $CKPT_DIR/humaneval_metrics_temperature_sampling_${t}_ranked_pass_eval_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.txt &
            # exec@k, ranked exec@k evaluation
            python human_eval/evaluate_functional_correctness.py \
                --task humaneval \
                --sample_file $CKPT_DIR/humaneval_samples_temperature_sampling_${t}_ranked_eval_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.json \
                --score_field mean_logp \
                --k '1,5,10,100' \
                --exec_check=True \
                > $CKPT_DIR/humaneval_metrics_temperature_sampling_${t}_ranked_exec_eval_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.txt &
        else
            # pass@k
            python human_eval/evaluate_functional_correctness.py \
                --task humaneval \
                --sample_file $CKPT_DIR/humaneval_samples_temperature_sampling_${t}_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.json \
                --k '1,5,10,100' \
                --exec_check=False \
                > $CKPT_DIR/humaneval_metrics_temperature_sampling_${t}_pass_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.txt &
            # exec@k
            python human_eval/evaluate_functional_correctness.py \
                --task humaneval \
                --sample_file $CKPT_DIR/humaneval_samples_temperature_sampling_${t}_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.json \
                --k '1,5,10,100' \
                --exec_check=True \
                > $CKPT_DIR/humaneval_metrics_temperature_sampling_${t}_exec_eval_max_length_${MAX_COMP_LENGTH}_samples_${NUM_SAMPLES}.txt &
        fi
    done
    wait

done
