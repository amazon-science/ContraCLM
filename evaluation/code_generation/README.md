# Evaluation on Code Generation

This folder contains scripts for performing execution based evaluation of code generation models on the HumanEval and MBPP benchmarks.

### File Descriptions

|File/Folder Name|Description|
|----|----|
|`execution_utils.py`| Helper script for running HumanEval and MBPP. Well maintained and documented file.|
|`humaneval.py`| Driver script for HumanEval. Load data -> run evaluation.|
|`human_eval/`| Directory containing a modified version of `human_eval` library that supports pass@k, ranked pass@k, exec@k, ranked exec@k.|

### Prerequisites

1. Make sure you have the `human_eval` folder
2. Download the HumanEval data into `data/` following the instructions given in `data/`.


### Decoding Strategies

The following decoding strategies are currently supported. Pass the relevant arguments that are required for each decoding strategy as indicated below.

```python
if args.decoding_strategy == "greedy":
    decoding_strategy_kwargs = {'do_sample': False, 'num_beams': 1}

elif args.decoding_strategy == "nucleus":
    decoding_strategy_kwargs = {'do_sample': True, 'top_p': args.top_p, 
                                'top_k': 0,
                                'temperature': args.sampling_temperature}

elif args.decoding_strategy == "temperature_sampling":
    decoding_strategy_kwargs = {'do_sample': True, 'top_k': 0, 
                                'temperature': args.sampling_temperature}
```

Refer [HF documentation link 1](https://huggingface.co/blog/how-to-generate) and [HF documentation link 2](https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate) for more details. 


### HumanEval

**Loading HumanEval data**

```python
from human_eval.data import read_problems
problems = read_problems()
```

**Usage**

1. **Checkpoint location:** Running this evaluation requires specifying the exact path to the checkpoint that needs to be evaluated. Specify this using `CKPT_DIR` and `MODEL_PATH` inside `humaneval.sh`.

2. **Parallel evaluation:** The script is designed to run 4 temperature evaluations in parallel on 4 different GPUs on the same machine. These are controlled using `temperatures` and `devices` inside `humaneval.sh`. Modify according to need and device availability. *Pro-tip: Use (0 1 2 3) for one checkpoint and (4 5 6 7) for another checkpoint on the same machine with 8 GPUs*.

3. Run: `bash humaneval.sh`

For more information, see the explanation of a single run below.

**Usage: single run**

Some of the arguments/flags are explained below. For concrete example refer `humaneval.sh`.

```bash
DECODING='--decoding_strategy temperature_sampling --sampling_temperature '$T

python humaneval.py \
    --model_name Salesforce/codegen-350M-mono \
    --pad_token_id 50256 \
    --ckpt_dir $CKPT_DIR \ # directory of model that needs to be evaluated
    --model_path $MODEL_PATH \ # relative path to checkpoint inside $CKPT_DIR
    $PRETRAINED_EVAL \ # '--eval_pretrained' if want to evaluate pretrained else ''
    --seed 1234 \
    --accelerator gpu \
    $DECODING \
    --eval_device $DEV \ # will use f'cuda:{eval_device}' for evaluation
    --deepspeed_ckpt \
    $RANKED_EVAL \ # '--ranked_pass_at_k' if want to calculate ranked metrics else ''
    --num_samples_per_task $NUM_SAMPLES \
    --max_completion_length $MAX_COMP_LENGTH \
    --tok_max_length $TOK_MAX_LEN \
    --batch_cnt $BATCH_CNT # refer args help for description of these
```

This will generate a single `.json` file at `$CKPT_DIR` containing the generated sequences. Let's call this file `generated.json` (the actual name of this file is determined by `out_file`. See `execution_utils.setup_filename`). We can compute the metrics now as:

```bash
# pass@k, ranked pass@k evaluation
python human_eval/evaluate_functional_correctness.py --task humaneval --sample_file $CKPT_DIR/generated.json --score_field mean_logp --k '1,5,10,100' --exec_check=False \
    > log1
    
# exec@k, ranked exec@k evaluation
python human_eval/evaluate_functional_correctness.py --task humaneval --sample_file $CKPT_DIR/generated.json --score_field mean_logp --k '1,5,10,100' --exec_check=True \
    > log2
```

The `python` command prints the metrics to the terminal. We direct the output to the files `log1` and `log2`. These names are used only for illustration. See `humaneval.sh` for more concrete examples. *Pro-tip: follow how these files are custom-named for different configurations in `humaneval.sh`.*
