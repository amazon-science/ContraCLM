LR=2e-5
GRAD_STEPS=1
TRAIN_BS=512
VALID_BS=64
EPOCHS=12
WARMUP_STEPS=500
MAX_STEPS=-1
GPUS=0,1,2,3,4,5,6,7
GPU_COUNT=8
NUM_WORKS=8

TEMP=0.05
MARGIN=0.5
DROPOUT_P=0.1

## Augmented data w/ deduplication
DATA_DIR=/path/to/data/dir
TRAIN_DIR="${DATA_DIR}/rel/path/to/train/data"
VALID_DIR="${DATA_DIR}/rel/path/to/val/data"

options=(
   '--loss MLE_Only'
   '--loss ContraCLMSeq --temperature $TEMP'
   '--loss ContraCLMTok --temperature $TEMP'
   '--loss ContraCLM --temperature $TEMP'
)

export CUDA_HOME=/usr/local/cuda

for method in 3
do
    CL_Config=$(eval echo ${options[$method]})

    CUDA_VISIBLE_DEVICES=$GPUS python pl_trainer.py \
        --num_workers $NUM_WORKS \
        --devices $GPU_COUNT \
        --accelerator gpu \
        --model_name gpt2 \
        --pad_token_id 50256 \
        --dropout_p $DROPOUT_P \
        --expt_prefix wikitext_103 \
        --default_root_dir ./logs_store/deepspeed/ \
        --train_datadir $TRAIN_DIR \
        --valid_datadir $VALID_DIR \
        --log_dir logs \
        --seed 42 \
        --lr $LR \
        --weight_decay 0.1 \
        --gradient_clip_val 1.0 \
        --max_epochs $EPOCHS \
        --max_steps $MAX_STEPS \
        --warmup_steps $WARMUP_STEPS \
        --train_batch_size $TRAIN_BS \
        --valid_batch_size $VALID_BS \
        --accumulate_grad_batches $GRAD_STEPS \
        --log_every_n_steps 100 \
        --save_step_frequency 1000 \
        --val_check_interval 1000 \
        --debug_cuda_mem \
        --use_deepspeed \
        --precision 16  \
        $CL_Config
    wait
done