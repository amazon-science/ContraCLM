DATA_DIR="/path/to/data/dir"
OUTPUT_DIR="/path/to/output/dir"
SEQ_LENGTH=512

python preprocess_wikitext.py \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --seq_length $SEQ_LENGTH
