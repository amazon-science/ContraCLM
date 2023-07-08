# Text Discrimination Evaluation

### STS

Dataset: download the dataset by runing the following: 
```bash
./SentEval/data/downstream/download_dataset.sh
```
   
### Single run command 
```bash
python eval_sts.py \
    --path_to_sts_data ./SentEval/data \
    --pretrained_model GPT2 \
    --gpt gpt2 \
    --respath ./results/ \
    --eval_deepspeed_ckpt \
    --device_id 2
```
