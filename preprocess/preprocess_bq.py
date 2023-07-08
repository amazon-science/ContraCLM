'''
This file reads JSONL files of raw code data and splits the entire dataset into
train/valid/test splits and stores as arrow datasets.

This script also chunks and tokenizes train, validation, test splits of the arrow
dataset. The information stored in the output dataset contains the chunked string,
location of original string, and information like chunk index to uniquely
identify each chunk.

Finally, this script performs deduplication of chunked dataset based on chunk
exact match (EM).
'''
import argparse
import glob
import json
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import AutoTokenizer

args, tokenizer = None, None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
        help='Directory to the data directory containing JSONL files with raw data')
    parser.add_argument('--output_dir', type=str, 
        help='Directory to store preprocessed dataset')
    parser.add_argument('--seed', type=int, default=0, 
        help='Seed value to use while shuffling in splitting')
    parser.add_argument('--test_and_valid_combined_size', type=float,
        default=0.004, help='% value of how much data to use for valid+test')
    parser.add_argument('--seq_length', type=int, default=512, 
        help='Maximum sequence length while tokenizing')
    parser.add_argument('--chars_per_tok', type=float, default=3.2, 
        help='Number of characters per token, on average')
    parser.add_argument('--codegen_pad_token_id', type=int, default=50256, 
        help='Padding token ID for CodeGen tokenizer')
    args = parser.parse_args()
    args.max_chars_per_seq = args.seq_length * args.chars_per_tok
    return args


def process_bq(bq_dir):
    '''Funtion to read the BigQuery dataset'''
    # get all jsonl files
    all_files = glob.glob(os.path.join(bq_dir, '*.jsonl'))
    # read all jsonl files and collect all data together tracking origin files
    all_data = {'content': [], 'origin': [], 'origin_index': []}
    print(f'Reading raw BigQuery data...')
    for filepath in tqdm(all_files):
        origin = '/'.join(filepath.split('/')[-2:])
        with open(filepath, 'r') as f:
            for line_num, line in enumerate(f):
                raw_str = json.loads(line)['content']
                all_data['content'].append(raw_str)
                all_data['origin'].append(origin)
                all_data['origin_index'].append(line_num)
    return all_data


def dataset_from_all_data(all_data, seed, test_and_valid_combined_size):
    '''Split `all_data` into train/valid/test splits'''
    dataset = Dataset.from_dict(all_data)
    train_testvalid = dataset.train_test_split(
        test_size=test_and_valid_combined_size, shuffle=True, seed=seed
    )
    test_valid = train_testvalid['test'].train_test_split(
        test_size=0.5, shuffle=True, seed=seed
    )
    split_dataset = DatasetDict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']
    })
    return split_dataset


def chunk(raw_str, max_chars_per_seq):
    '''Chunk `raw_str` such that each chunk ends at a newline.
    Also ensures each chunk has <= `max_chars_per_seq` characters'''
    all_lines = raw_str.splitlines(keepends=True)
    cstr = all_lines[0]
    chunked_strs = []
    last_line_nums = []
    for i, line in enumerate(all_lines[1:], start=1):
        if len(cstr) + len(line) <= max_chars_per_seq:
            cstr += line # continue appending till space is remaining
        else:
            chunked_strs.append(cstr)
            last_line_nums.append(i - 1)
            cstr = line
    chunked_strs.append(cstr)
    last_line_nums.append(len(all_lines) - 1)
    assert ''.join(chunked_strs) == raw_str
    return {
        'content': chunked_strs, 
        'chunk_index': list(range(len(chunked_strs))), 
        'last_line_num': last_line_nums
    }


def chunk_and_tokenize(batch):
    global tokenizer, args
    ret_dict = {
        'content': [],
        'origin': [],
        'origin_index': [],
        'token_ids': [],
        'chunk_index': [],
        'last_line_num': []
    }
    batch_size = len(batch['content'])

    for i in range(batch_size):
        complete_code = batch['content'][i]
        origin = batch['origin'][i]
        origin_index = batch['origin_index'][i]

        # chunk
        chunked_dict = chunk(complete_code, args.max_chars_per_seq)
        # tokenize
        token_ids = tokenizer(
            chunked_dict['content'], padding='max_length', 
            truncation=True, max_length=args.seq_length
        ).input_ids

        # update vals to return
        chunk_cnt = len(chunked_dict['content'])
        ret_dict['content'] += chunked_dict['content']
        ret_dict['origin'] += [origin] * chunk_cnt
        ret_dict['origin_index'] += [origin_index] * chunk_cnt
        ret_dict['token_ids'] += token_ids
        ret_dict['chunk_index'] += chunked_dict['chunk_index']
        ret_dict['last_line_num'] += chunked_dict['last_line_num']

    return ret_dict


def deduplicate(dataset):
    '''Deduplicate dataset based on EM of sequences'''
    print('Reading strings from dataset for deduplication...', end='', flush=True)
    # reading entire dataset can be time consuming due to token ids
    # so first, get dedup indexes using only strings
    strs = dataset['content']
    print('Done.', flush=True)
    ser = pd.Series(strs)
    dedups = ser.drop_duplicates()
    doc_ids_to_retain = dedups.index.tolist()
    print(f'Before: {len(dataset)}, After: {len(doc_ids_to_retain)}')
    # now read entire dataset and filter based on above findings
    print('Filtering entire dataset...')
    filtered_data = dataset.select(doc_ids_to_retain, keep_in_memory=True)
    return filtered_data


def main():
    global args, tokenizer
    args = get_args()
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
    tokenizer.pad_token_id = args.codegen_pad_token_id

    # gather all data from desired data version
    all_data = process_bq(args.data_dir)
    # split data
    dataset = dataset_from_all_data(
        all_data, args.seed, args.test_and_valid_combined_size
    )

    # chunk and tokenize
    print(f'Chunking to sequence length {args.seq_length}')
    dataset = dataset.map(chunk_and_tokenize, batched=True)

    # deduplicate
    for split in ['train', 'valid', 'test']:
        print('Deduplicating', split, '...')
        dataset[split] = deduplicate(dataset[split])

    # save data
    dataset.save_to_disk(args.output_dir)


if __name__ == '__main__':
    main()
