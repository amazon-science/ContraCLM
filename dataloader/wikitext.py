import os
import random
import torch
import numpy as np
import jsonlines
from transformers import GPT2Tokenizer, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk

import logging
logger = logging.getLogger(__name__)

class TextLLMRetrievalDataset(Dataset):
    def __init__(
        self,
        tokenizer=None,
        data_path=None,
        cache_path=None,
        max_len=256,
        num_neighbors=4,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.cache_path = cache_path
        self.max_len = max_len
        self.num_neighbors = num_neighbors

        if not os.path.exists(cache_path):
            assert data_path != None, "data_path cannot be None when there is no cached data!!!"
            TextLLMRetrievalDataset.preprocess_and_cache(tokenizer, data_path, cache_path, num_neighbors, max_len)

        logger.info("Loading cache data...")
        self.data = load_from_disk(cache_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        source_tokens = np.array(self.data[ind]["input_ids"])
        source_tokens_torch = torch.from_numpy(source_tokens).long()

        if self.num_neighbors > 0:
            retrieval_tokens = np.array(self.data[ind]["input_ids_ret"])
            retrieval_tokens_torch = torch.from_numpy(retrieval_tokens).long()
            return {"input_ids": source_tokens_torch, "input_ids_ret": retrieval_tokens_torch}
        else:
            return {"input_ids": source_tokens_torch}

    @classmethod
    def preprocess_and_cache(cls, tokenizer, data_path, cache_path, num_neighbors, max_len):
        from datasets import Dataset
        if os.path.exists(cache_path):
            print("Found cache files. Reusing cache data file")
            return

        raw_data = list(jsonlines.open(data_path))
        print(f"preprocess and cache data from: \t {data_path} with {len(raw_data)} examples")
        process_dict = {
            'content': [],
            'input_ids': [],
            'data_idx': [],
            'ret_idx': [],
            'input_ids_ret': [],
        }
        for idx, ex in enumerate(raw_data):
            input_str = ex['input_str']
            input_ids = tokenizer(input_str, padding='max_length', truncation=True, max_length=max_len)["input_ids"]

            if num_neighbors > 0:
                if not len(ex["retrieved_neighbors"]) in [num_neighbors, num_neighbors-1]:
                    real_neighbors = len(ex["retrieved_neighbors"])
                    print(f"sample-{idx}-with-{real_neighbors} neighbors")
                    continue
                retrieval = ex["retrieved_neighbors"]
                retrieval_ids = tokenizer(retrieval, padding='max_length', truncation=True, max_length=max_len)["input_ids"]

            process_dict['data_idx'].append(idx)
            process_dict['content'].append(input_str)
            process_dict['input_ids'].append(input_ids)
            process_dict['ret_idx'].append(ex["retrieved_index"])
            process_dict['input_ids_ret'].append(retrieval_ids)

        print(len(process_dict['input_ids']), len(process_dict['input_ids_ret']), len(retrieval_ids), len(process_dict['input_ids_ret'][0]))
        print(f"All Examples: {len(raw_data)}, {len(process_dict['input_ids'])} with valid neighbors")

        print("writing to cache")
        tokenized_dataset = Dataset.from_dict(process_dict)
        tokenized_dataset.save_to_disk(cache_path)
        return

def preprocess_data():
    data_dir = "/path/to/data/dir"
    data_name = "wikitext_103_sparse_retrieval"
    num_neighbors = 32
    source_path = data_dir + f"{data_name}_k{num_neighbors}_uniq.jsonl"
    cache_path = data_dir + f"{data_name}_k{num_neighbors}_cached_train_uniq"

    if not os.path.exists(cache_path):
        # # tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = 50256
        dataset = TextLLMRetrievalDataset(tokenizer, source_path, cache_path, 256, num_neighbors)
        idx = 10
        print(dataset[idx].keys(), len(dataset[idx]["input_ids"]), len(dataset[idx]["retrieval"]), len(dataset[idx]["retrieval"][0]))

    # train_data = load_from_disk(cache_path)
    # train_data.set_format(type='torch', columns=['input_ids', 'input_ids_ret'])
    train_data = TextLLMRetrievalDataset(None, None, cache_path, 256, num_neighbors)
    dl = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=2)
    for i, batch in enumerate(dl):
        index, input_ids, input_ids_ret = batch["idx"], batch['input_ids'], batch['input_ids_ret']
        print("input_ids \t", index, type(input_ids), input_ids.size(), input_ids_ret.size())
        break
    print("Done")

if __name__ == "__main__":
    preprocess_data()

