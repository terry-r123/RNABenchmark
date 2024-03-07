import warnings

warnings.filterwarnings("ignore")
import os
import numpy as np
import torch
from torch.utils.data import Dataset

def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

class SSDataset(Dataset):
    def __init__(self, df, data_path, tokenizer, args=None):
        self.df = df
        self.data_path = data_path
        self.tokenizer = tokenizer
        print(f'len of dataset: {len(self.df)}')
        if args:
            self.args = args
        token_test = df.iloc[0]['seq'].replace('U', 'T')
        if self.args.token_type == '6mer':
            token_test = generate_kmer_str(token_test, 6)
        print(token_test)
        print(tokenizer.tokenize(token_test))
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        seq = seq.replace('U', 'T')
        # if self.args.token_type == '6mer':
        #     seq = generate_kmer_str(seq, 6)
            #print(seq)
        if self.args.model_type == 'esm-protein':
            seq = generate_protein(seq)
        file_name = row['file_name']
        
        file_path = os.path.join(self.data_path, file_name + '.npy')
        os.path.exists(file_path)
  
        #print(file_path)
        ct = np.load(file_path)
        #print(seq,ct)
        return seq, ct, self.args

