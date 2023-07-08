LR=2e-5
GRAD_STEPS=16
TRAIN_BS=32
VALID_BS=256
EPOCHS=3
STEPS=-1
WARMUP_STEPS=500
GPUS=0,1,2,3,4,5,6,7
GPU_COUNT=8
NODE_COUNT=1
NUM_WORKERS=96

TEMP=0.05
MARGIN=0.5

CODEGEN_DROPOUT_LAYERS=0
CODEGEN_DROPOUT_P=0.1

DATA_DIR="/path/to/data/dir"
TRAIN_DIR="${DATA_DIR}/bigquery_pypi_chunked_512/train"
VALID_DIR="${DATA_DIR}/bigquery_pypi_chunked_512/valid"

export CUDA_HOME=/usr/local/cuda

options=(
   '--loss MLE_Only'
   '--loss ContraCLMSeq --temperature $TEMP'
   '--loss ContraCLMTok --temperature $TEMP'
   '--loss ContraCLM --temperature $TEMP'
)

for ind in 3
do
   CL=$(eval echo ${options[$ind]})
   ### with retrieval data
   CUDA_VISIBLE_DEVICES=$GPUS python pl_trainer.py \
      --num_workers $NUM_WORKERS \
      --devices $GPU_COUNT \
      --accelerator gpu \
      --model_name Salesforce/codegen-350M-mono \
      --pad_token_id 50256 \
      --dropout_layers $CODEGEN_DROPOUT_LAYERS \
      --functional_dropout \
      --dropout_p 0.1 \
      --expt_prefix BigQuery_v3 \
      --default_root_dir ./logs_store/deepspeed/ \
      --train_datadir $TRAIN_DIR \
      --valid_datadir $VALID_DIR \
      --log_dir ./logs/ \
      --seed 1234 \
      --lr $LR \
      --weight_decay 0.1 \
      --gradient_clip_val 1.0 \
      --max_steps $STEPS \
      --max_epochs $EPOCHS \
      --warmup_steps $WARMUP_STEPS \
      --train_batch_size $TRAIN_BS \
      --valid_batch_size $VALID_BS \
      --accumulate_grad_batches $GRAD_STEPS \
      --log_every_n_steps 50 \
      --save_step_frequency 500 \
      --val_check_interval 250 \
      --debug_cuda_mem \
      --use_deepspeed \
      --precision 16  \
      $CL

   wait
done
