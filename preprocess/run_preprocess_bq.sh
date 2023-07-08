DATA_DIR="/path/to/data/dir"
OUTPUT_DIR="/path/to/output/dir"
SEQ_LENGTH=512

python preprocess_bq.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --seed 0 \
    --test_and_valid_combined_size 0.004 \
    --seq_length $SEQ_LENGTH \
    --chars_per_tok 3.2 \
    --codegen_pad_token_id 50256
