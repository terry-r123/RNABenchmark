import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import csv
import pandas as pd
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

class SSDataset(Dataset):
    def __init__(self, data_path, tokenizer, args, mode):
        df = pd.read_csv(f'{data_path}/bpRNA.csv')
        if mode=='train':
            df = df[df['data_name'] == 'TR0'].reset_index(drop=True)
            data_path=f'{data_path}/TR0'
        elif mode=='val':
            df = df[df['data_name'] == 'VL0'].reset_index(drop=True)
            data_path=f'{data_path}/VL0'
        elif mode=='test':
            df = df[df['data_name'] == 'TS0'].reset_index(drop=True)
            data_path=f'{data_path}/TS0'
        self.num_labels = 1
        self.df = df
        self.data_path = data_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')       
        self.args = args

        token_test = df.iloc[0]['seq'].upper().replace("U", "T")
        if 'mer' in self.args.token_type:
            token_test = generate_kmer_str(token_test, int(self.args.token_type[0]))
        print(token_test)
        test_example = tokenizer.tokenize(token_test)
        print(test_example)
        print(tokenizer(token_test))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.upper().replace("U", "T")
        # if self.args.model_type == 'esm-protein':
        #     seq = generate_protein(seq)
        file_name = row['file_name']
        
        file_path = os.path.join(self.data_path, file_name + '.npy')
        os.path.exists(file_path)
  
        #print(file_path)
        struct = np.load(file_path)
        #print(seq,ct)
        return dict(seq=seq, struct=struct)

class ContactMapDataset(Dataset):
    def __init__(self, data_path, tokenizer, args):
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[1].upper().replace("U", "T") for d in data]
            ids = [(d[0]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        #texts = [generate_kmer_str(text) for text in texts]
        self.tokenizer = tokenizer
        self.args = args
        self.ids = ids
        # Turn the text into input_ids by
        self.texts = texts
        self.num_labels = 1
        self.data_path = data_path
        # target path is in the same directory as the text file
        parent_dir = os.path.dirname(data_path)
        self.target_path = os.path.join(parent_dir, "contact_map")
        # ensure tokenier
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        id  = self.ids[idx]
        target_path = self.target_path + "/" + id + ".npy"
        struct = np.load(target_path).astype(float) 
        seq = self.texts[idx]
        if len(seq) > self.tokenizer.model_max_length-2:
            seq = seq[:self.tokenizer.model_max_length-2]
            struct = struct[:self.tokenizer.model_max_length-2][:self.tokenizer.model_max_length-2]
        return dict(seq=seq, struct=struct)

class DistanceMapDataset(Dataset):
    def __init__(self, data_path, tokenizer, args):
        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            texts = [d[1].upper().replace("U", "T") for d in data]
            ids = [(d[0]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        #texts = [generate_kmer_str(text) for text in texts]
        self.tokenizer = tokenizer
        self.args = args
        self.ids = ids
        # Turn the text into input_ids by
        self.texts = texts
        self.num_labels = 1
        self.data_path = data_path
        # target path is in the same directory as the text file
        parent_dir = os.path.dirname(data_path)
        self.target_path = os.path.join(parent_dir, "distance_map")
        # ensure tokenier
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        id  = self.ids[idx]
        target_path = self.target_path + "/" + id + ".npy"
        struct = np.load(target_path).astype(float) 
        seq = self.texts[idx]
        if len(seq) > self.tokenizer.model_max_length-2:
            seq = seq[:self.tokenizer.model_max_length-2]
            struct = struct[:self.tokenizer.model_max_length-2][:self.tokenizer.model_max_length-2]
        return dict(seq=seq, struct=struct)


