import argparse
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    GPT2LMHeadModel,
    BartForConditionalGeneration,
    BartTokenizer,
    T5Tokenizer,
    T5ForConditionalGeneration
)

import sys
sys.path.append("../../../")

BERT_CLASS = {
    "bertbase": 'bert-base-uncased',
    "bertlarge": 'bert-large-uncased',
    "gpt2": "gpt2",
    "pairsupcon-base": "aws-ai/pairsupcon-bert-base-uncased",
    "pairsupcon-large": "aws-ai/pairsupcon-bert-large-uncased",
}

def get_args(argv):
    parser = argparse.ArgumentParser("Evaluation of the pre-trained models on various downstream tasks")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--seed', type=int, default=0)
    # results path and prefix
    parser.add_argument('--respath', type=str, default="../../downstream_evalres/senteval/")
    # data evaluation configuration
    parser.add_argument('--path_to_sts_data', type=str, default="", help="path to the SentEval data")
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    # evaluate the pretrained_model
    parser.add_argument('--pretrained_dir', type=str, default="")
    parser.add_argument('--pretrained_model', type=str, default='GPT2', choices=["BERT", "SBERT", "PairSupCon", "GPT2", "BART", "T5"])
    parser.add_argument('--bert', type=str, default='bertbase')
    parser.add_argument('--gpt', type=str, default='gpt2')
    parser.add_argument('--pretrain_batchsize', type=int, default=512)
    parser.add_argument('--ckpt_step', type=int, default=0)
    parser.add_argument('--pad_token_id', type=int, default=50256)
    parser.add_argument("--eval_deepspeed_ckpt", action="store_true",
                        help="If True, will attempt to load a checkpoint saved while training with deepspeed")
    args = parser.parse_args(argv)
    return args

def get_bert(args):
    config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
    model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])

    return model, tokenizer

def get_gpt():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    return model, tokenizer


def get_checkpoint(args):
    if args.pretrained_model in ["BERT", "PairSupCon"]: # evaluate vanilla BERT or PairSupCon
        model, tokenizer = get_bert(args)
        print("...... loading BERT", args.bert, "resname ", resname)
    elif args.pretrained_model == "BART":
        model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", forced_bos_token_id=0)
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    elif args.pretrained_model == "T5":
        tokenizer = T5Tokenizer.from_pretrained("t5-base")
        model = T5ForConditionalGeneration.from_pretrained("t5-base")

    elif args.pretrained_model == "GPT2": # evaluate SentenceBert
        model, tokenizer = get_gpt()
        print("...... loading GPT2--", args.gpt, "resname ", resname)
        # model.resize_token_embeddings(len(tokenizer))

    else: 
        raise Exception("please specify the pretrained model you want to evaluate")
             
    model.to(args.device)
    model.eval()
    return model, tokenizer, args.pretrained_model

