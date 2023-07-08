import argparse
import os

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer


def config_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=None, 
                        help='Directory to load the raw WikiText-103 dataset from')
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Directory to store preprocessed dataset')
    parser.add_argument('--seq_length', type=int, default=256, 
                        help='Maximum sequence length while tokenizing')
    parser.add_argument('--chars_per_tok', type=float, default=3.2, 
                        help='Number of characters per token, on average')
    parser.add_argument('--pad_token_id', type=int, default=50256, 
                        help='Padding token ID')
    parser.add_argument("--model_name", type=str, default='gpt2', 
                        choices=["gpt2"])
    args = parser.parse_args()
    args.max_chars_per_seq = args.seq_length * args.chars_per_tok
    return args


def truncate_and_tokenize(args, tokenizer, text_data):
    ret_dict = {
        'content': [],
        'input_ids': [],
        'attention_mask': [],
        'data_idx': [],
    }
    for data_idx, txt in tqdm(enumerate(text_data), total=len(text_data)):
        features = tokenizer(txt,
                             padding='max_length',
                             truncation=True,
                             max_length=args.seq_length)
        ret_dict['content'] += [txt]
        ret_dict['input_ids'] += [features['input_ids']]
        ret_dict['attention_mask'] += [features['attention_mask']]
        ret_dict['data_idx'].append(data_idx)
    print(len(text_data), ret_dict.keys(), 
          len(ret_dict['content']), len(ret_dict['input_ids']))
    return ret_dict


def tokenize_wikitext_103(args):
    for datatype in ['train', 'validation', 'test']:
        output_path = os.path.join(
            args.output_dir, f"wikitext103_raw_v1_{datatype}"
        )
        data_path = os.path.join(
            args.data_dir, f"wikitext103_raw_v1_{datatype}"
        )
        os.makedirs(output_path, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token

        dataset = load_dataset("text", data_files=data_path + ".txt")
        print(type(dataset), len(dataset), dataset.keys(), 
              len(dataset[datatype]["text"]))

        tokenized_dict = truncate_and_tokenize(args, tokenizer, dataset[datatype]["text"])
        tokenized_dataset = Dataset.from_dict(tokenized_dict)
        tokenized_dataset.save_to_disk(output_path)


def merge_data(args):
    datadict = {}
    for datatype in ["validation", "test", "train"]:
        data_path = os.path.join(args.output_dir, f"wikitext103_raw_v1_{datatype}")
        dataset = load_from_disk(data_path)
        print(type(dataset), len(dataset))
        if datatype == "validation":
            datadict["valid"] = dataset
        else:
            datadict[datatype] = dataset
    merged_dataset = DatasetDict(datadict)
    os.makedirs(args.output_dir, exist_ok=True)
    merged_dataset.save_to_disk(
        os.path.join(args.output_dir, f"wikitext103_raw_v1_tokenized_all")
    )


if __name__ == "__main__":
    args = config_args()
    tokenize_wikitext_103(args)
    merge_data(args)
