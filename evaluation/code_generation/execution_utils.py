'''This file contains utility functions specifically for execution based 
evaluation of code generation models'''

import argparse
import os
import sys

sys.path.append('../../')
import torch
from tqdm import tqdm

from pl_model import LitContraCLM
from utils import load_model_and_tokenizer


def eval_args():
    '''Set and parse arguments specific to execution-based 
    generation evaluation'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", type=str, 
        help="device to run generations on", choices=["gpu", "cpu"], 
        default="cpu")
    parser.add_argument("--seed", type=int, default=42, 
        help="value to seed RNG of torch, numpy")
    parser.add_argument("--model_name", type=str, 
        choices=["Salesforce/codegen-350M-mono"])
    parser.add_argument("--pad_token_id", type=int, default=50256)
    parser.add_argument("--model_path", type=str, 
        help="Relative path to the model to evaluate")
    parser.add_argument("--trainer_args_path", type=str, 
        help="Relative path to the pl.Trainer args corresponding to checkpoint")
    parser.add_argument("--ckpt_dir", type=str, 
        help="Path to the directory with model checkpoints and outputs")
    parser.add_argument("--decoding_strategy", type=str, 
        choices=["greedy", "nucleus", "temperature_sampling"], default="greedy", 
        help="Type of decoding while generating sequence using LM")
    parser.add_argument("--sampling_temperature", type=float, default=1.0, 
        help="Temperature value for temperature based sampling while decoding")
    parser.add_argument("--top_p", type=float, default=1.0, 
        help="Probability value threshold in nucleus sampling")
    parser.add_argument("--eval_pretrained", action="store_true", 
        help="Flag to indicate if to evaluate the original pretrained model")
    parser.add_argument("--mbpp_problems_path", type=str, 
        help="Path to problems of the MBPP benchmark")
    parser.add_argument("--eval_device", type=int, default=0, 
        help="Index of GPU to use while evaluation")
    parser.add_argument("--deepspeed_ckpt", action="store_true", 
        help="If True, will attempt to load a checkpoint saved while "
        "training with deepspeed")
    parser.add_argument("--ranked_pass_at_k", action="store_true", 
        help="If True, will perform ranked pass@k evaluation")
    parser.add_argument("--max_completion_length", type=int, default=256,
        help="Maximum number of tokens to generate after the prompt")
    parser.add_argument("--num_samples_per_task", type=int, default=200,
        help="Number of sequences to generate for each prompt")
    parser.add_argument("--tok_max_length", type=int, default=1024,
        help="Maximum number of tokens in the prompt before left-truncation "
        "is performed")
    parser.add_argument("--batch_cnt", type=int, default=4,
        help="# batches over which to generate num_samples_per_task sequences")
    args = parser.parse_args()
    return args


def setup_filename(args, task_name):
    out_file = f'{task_name}_samples'
    if args.decoding_strategy in ['temperature_sampling', 'nucleus']:
        out_file += f'_{args.decoding_strategy}_{args.sampling_temperature}'
    else:
        out_file += f'_{args.decoding_strategy}'
    if args.ranked_pass_at_k:
        out_file += '_ranked_eval'
    out_file += f'_max_length_{args.max_completion_length}'
    out_file += f'_samples_{args.num_samples_per_task}'
    out_file += '.json'
    out_file = os.path.join(args.ckpt_dir, out_file)
    return out_file


def solve_tasks(problems, task_ids, model, tokenizer, args, decoding_kwargs):
    '''Generate solutions for `problems` using `model` based on `args`.
    Returns list of dicts that can be used by `evaluate_functional_correctness
    Args:
    problems: list of (str) prompts
    task_ids: list of (str) task identifiers in the dataset
    args: arguments passed during configuration parsing
    decoding_kwargs: (dict) object returned by `decoding_strategy` in this file'''
    samples = []
    for task_id, problem_str in tqdm(zip(task_ids, problems), total=len(task_ids)):
        if not args.ranked_pass_at_k:
            completions = []
            for _ in range(args.batch_cnt):
                completions += generate_n_completions(
                    problem_str,
                    tokenizer,
                    model,
                    decoding_kwargs,
                    n=args.num_samples_per_task // args.batch_cnt,
                    tok_max_length=args.tok_max_length,
                    max_completion_length=args.max_completion_length
                ) # to avoid GPU oom
            samples += [
                dict(task_id=task_id, completion=completion) 
                for completion in completions
            ]
        else:
            completions, mean_logps = [], []
            for _ in range(args.batch_cnt):
                t_completions, t_mean_logps = generate_n_ranked_completions(
                    problem_str,
                    tokenizer,
                    model,
                    decoding_kwargs,
                    n=args.num_samples_per_task // args.batch_cnt,
                    tok_max_length=args.tok_max_length,
                    max_completion_length=args.max_completion_length
                ) # to avoid GPU oom
                completions += t_completions
                mean_logps += t_mean_logps
            samples += [
                dict(task_id=task_id, completion=completion, mean_logp=mlp) 
                for completion, mlp in zip(completions, mean_logps)
            ]
    return samples



def get_single_indent_function_by_indent(completion):
    '''Truncate `completion` to the first complete function 
    with a single level of indent'''
    # all prompts in human-eval/modified MBPP have single 
    # level of indent in the method to be completed, so set indent_level to 4
    indent_level = 4 
    valid_completions = []
    for line in completion.split("\n"):
	# elaborate conditions. could merge later.
        if line.strip().startswith('#'):
            '''
            take care of no-indent comments
            def xxx():
            # this is a function
                some code
            '''
            valid_completions.append(line)
        elif line == "":
            valid_completions.append(line)
        elif len(valid_completions) and valid_completions[-1].strip().endswith('\\'):
            '''
            take care of multi-line case
            a = '123' \
            '456'
            '''
            valid_completions.append(line)
        else:
            line = line.replace("\t", " " * 4)
            if not line.startswith(" " * indent_level):
                break
            valid_completions.append(line)
    valid_completion_str = "\n".join(valid_completions)
    return valid_completion_str


def load_deepspeed_state_dict(state_dict_path):
    '''Load the state of a torch model saved using deepspeed
    so that it can be easily loaded using torch.load'''
    state_dict = torch.load(state_dict_path, map_location="cpu")
    state_dict_attribute = None
    key_prefix = None
    if "state_dict" in state_dict:
        state_dict_attribute = "state_dict"
        key_prefix = "model"
    elif "module" in state_dict:
        state_dict_attribute = "module"
        key_prefix = "module.model"
    if state_dict_attribute:
        print(f"using state dict attribute {state_dict_attribute!r}")
        state_dict = state_dict[state_dict_attribute]
    if key_prefix:
        print(f"using state dict key prefix {key_prefix!r}")
        unwrapped_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(key_prefix):
                new_key = key[len(key_prefix) + 1 :]
                unwrapped_state_dict[new_key] = value
    else:
        unwrapped_state_dict = state_dict
    return unwrapped_state_dict


def generate_n_completions(prompt_str, tokenizer, model, 
                           decoding_strategy, n=1, tok_max_length=1024,
                           max_completion_length=256):
    '''prompt_str (str): single incomplete program that needs to be completed
    Complete prompt_str by generating `n` sequences according to 
    `decoding_strategy` and return complete functions'''
    device = model.device

    tokenizer_outs = tokenizer(
        [prompt_str], 
        padding=True, 
        return_tensors='pt', 
        truncation=True, 
        max_length=tok_max_length
    ) 
    (prompt_ids, prompt_attn_mask) = (
        tokenizer_outs.input_ids, 
        tokenizer_outs.attention_mask
    )
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids.to(device), 
            pad_token_id=tokenizer.pad_token_id,
            max_length=prompt_ids.size(1) + max_completion_length, 
            attention_mask=prompt_attn_mask.to(device),
            num_return_sequences=n,
            **decoding_strategy
        )
    pred_ids = generated_ids[:, prompt_ids.size(1):].cpu()
    pred_strs = [
        tokenizer.decode(pred_id, skip_special_tokens=True) 
        for pred_id in pred_ids
    ]
    pred_strs = [
        get_single_indent_function_by_indent(pred_str) 
        for pred_str in pred_strs
    ]
    return pred_strs


def generate_n_ranked_completions(prompt_str, tokenizer, model, 
                                  decoding_strategy, n=1, tok_max_length=1024,
                                  max_completion_length=256):
    '''prompt_str (str): single incomplete program that needs to be completed
    
    Complete prompt_str by generating `2*n` sequences according to 
    `decoding_strategy`, select top ranked `n` sequences (based on mean log(p)) 
    and return complete functions & probs for these `n` sequences'''
    device = model.device

    tokenizer_outs = tokenizer([prompt_str], 
                               padding=True, 
                               return_tensors='pt', 
                               truncation=True, 
                               max_length=tok_max_length) 
    (prompt_ids, prompt_attn_mask) = (
        tokenizer_outs.input_ids, 
        tokenizer_outs.attention_mask
    )
    with torch.no_grad():
        generated_ids = model.generate(
            prompt_ids.to(device), 
            pad_token_id=tokenizer.pad_token_id,
            max_length=prompt_ids.size(1) + max_completion_length, 
            attention_mask=prompt_attn_mask.to(device),
            num_return_sequences=n,
            **decoding_strategy
        )
    
    # get the trauncated sequences upto function completions
    pred_ids = generated_ids[:, prompt_ids.size(1):].cpu()
    pred_strs = [
        tokenizer.decode(pred_id, skip_special_tokens=True) 
        for pred_id in pred_ids
    ]
    pred_strs = [
        get_single_indent_function_by_indent(pred_str) 
        for pred_str in pred_strs
    ] # contains only the function definition

    # get the raw strings for complete functions (signature + definition)
    complete_function_strs = [prompt_str + pred_str for pred_str in pred_strs]
    # tokenize complete funcitons (note this is truncated output obtained above)
    complete_function_ids = [
        tokenizer(
            complete_function_str, 
            return_tensors='pt',
            truncation=True,
            max_length=prompt_ids.size(1) + max_completion_length
        ).input_ids[0]
        for complete_function_str in complete_function_strs
    ]

    # get logits for the truncated sequences using teacher forcing
    # cannot batch because each seq is of diff length and we don't pad
    with torch.no_grad():
        all_logits = [
            model(input_ids=cfi.to(device)).logits.cpu()
            for cfi in complete_function_ids
        ]
    # convert to probabilities
    complete_function_probs = [
        torch.nn.functional.softmax(logits, dim=-1)
        for logits in all_logits
    ]

    # complete_function_probs contains probabilities for entire vocab size 
    # we want the probs for only the predicted token 
    # whose indexes are in complete_function_ids
    next_token_probs = [
        torch.gather(
            complete_function_probs[i][:-1], 
            -1, 
            complete_function_ids[i][1:].unsqueeze(-1)
        ).squeeze() 
        for i in range(len(complete_function_ids))
    ] # input ids and logits are offset by 1, so slicing to match

    # ignore the predicted probabilities for the prompt
    next_token_probs = [
        ntp[prompt_ids.size(1) - 1:] 
        for ntp in next_token_probs
    ]

    # next_token_probs[i] is a tensor of size (indiv_completion_length)
    # so want to do: seq_probs[i] = geometric_mean(next_token_probs[i])
    # instead add log probabilities for numerical stability
    seq_probs = [torch.mean(torch.log(ntp)).item() for ntp in next_token_probs]

    return pred_strs, seq_probs


def load_ckpt_and_tokenizer(args, device):
    '''Load a checkpoint depending on requirements in `args`;
    loads model to device and sets in eval mode'''
    # load model
    if args.eval_pretrained:
        model, tokenizer = load_model_and_tokenizer(
            args.model_name, 
            pad_token_id=args.pad_token_id,
            dropout_layers=0
        )
    else:
        if args.deepspeed_ckpt:
            model, tokenizer = load_model_and_tokenizer(
                model_name=args.model_name, 
                pad_token_id=args.pad_token_id,
                dropout_layers=0
            )
            state_dict = load_deepspeed_state_dict(args.model_path)
            model.load_state_dict(state_dict)                                  
        else:
            trainer_args = Namespace(
                model_name=args.model_name, 
                pad_token_id=args.pad_token_id
            )
            pl_model = LitL2C.load_from_checkpoint(
                args.model_path, 
                trainer_args=trainer_args
            )
            model, tokenizer = pl_model.model, pl_model.tokenizer
    tokenizer.padding_side = 'left' # left padding while generation
    tokenizer.truncation_side = 'right' # remove suffix for very long sequences
    # TODO: This is originally 1024, check if this is alright to do
    tokenizer.model_max_length = 2048 
    model = model.to(device)
    model.eval()
    return model, tokenizer


def decoding_strategy(args):
    '''Get the kwargs for a decoding strategy specified in `args`
    The kwargs dict can be directly used while calling `generate` of HF model'''
    if args.decoding_strategy == "greedy":
        print(f'Sampling: t={args.sampling_temperature}')
        decoding_strategy_kwargs = {'do_sample': False, 'num_beams': 1}
        assert args.num_samples_per_task == 1, \
            'Attempting to generate > 1 sequences for greedy decoding'
    elif args.decoding_strategy == "nucleus":
        print(f'Nucleus: p={args.top_p},t={args.sampling_temperature}')
        decoding_strategy_kwargs = {'do_sample': True, 'top_p': args.top_p, 
                                    'top_k': 0,
                                    'temperature': args.sampling_temperature}
    elif args.decoding_strategy == "temperature_sampling":
        decoding_strategy_kwargs = {'do_sample': True, 'top_k': 0, 
                                    'temperature': args.sampling_temperature}
    return decoding_strategy_kwargs
