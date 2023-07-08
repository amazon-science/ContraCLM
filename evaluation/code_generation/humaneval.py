import os

import transformers
from execution_utils import (decoding_strategy, eval_args,
                             generate_n_completions,
                             generate_n_ranked_completions,
                             load_ckpt_and_tokenizer, setup_filename,
                             solve_tasks)
from human_eval.data import read_problems, write_jsonl
from packaging import version
from pytorch_lightning import seed_everything
from tqdm import tqdm


def main():
    # sanity checks
    assert version.parse(transformers.__version__) >= version.parse('4.21.0.dev0'), \
        "transformers version not supported" # critical for CodeGen
    
    # get config
    args = eval_args()
    seed_everything(args.seed, workers=True)
    device = f'cuda:{args.eval_device}' if args.accelerator == 'gpu' else 'cpu'

    # file paths
    if not args.eval_pretrained:
        assert args.model_path != None, 'Specify path to checkpoint'
        args.model_path = os.path.join(args.ckpt_dir, args.model_path)
    out_file = setup_filename(args, 'humaneval')

    # data
    problems = read_problems()
    prompts = [y['prompt'] for x, y in problems.items()]
    task_ids = list(problems.keys())

    # model and tokenizer in eval mode on device
    model, tokenizer = load_ckpt_and_tokenizer(args, device)

    # decoding strategy
    decoding_strategy_kwargs = decoding_strategy(args)

    # generate sequences
    samples = solve_tasks(
        prompts, task_ids, model, tokenizer, args, decoding_strategy_kwargs
    )
    write_jsonl(out_file, samples)


if __name__ == '__main__':
    main()


# References
# ----------
# https://github.com/openai/human-eval
