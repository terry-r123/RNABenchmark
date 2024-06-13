import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import random

import torch
import transformers
import sklearn
import scipy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import math
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_path)
sys.path.append(parent_dir)
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
from model.rnalm.modeling_rnalm import RnaLmForNucleotideLevel
from model.rnalm.rnalm_config import RnaLmConfig
from model.rnafm.modeling_rnafm import RnaFmForNucleotideLevel
from model.rnabert.modeling_rnabert import RnaBertForNucleotideLevel
from model.rnamsm.modeling_rnamsm import RnaMsmForNucleotideLevel
from model.splicebert.modeling_splicebert import SpliceBertForNucleotideLevel
from model.utrbert.modeling_utrbert import UtrBertForNucleotideLevel
from model.utrlm.modeling_utrlm import UtrLmForNucleotideLevel
from tokenizer.tokenization_opensource import OpenRnaLMTokenizer
early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})
    use_features: bool = field(default=True, metadata={"help": "whether to use alibi"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    tokenizer_name_or_path: Optional[str] = field(default="")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    data_train_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_val_path: str = field(default=None, metadata={"help": "Path to the training data."})
    data_test_path: str = field(default=None, metadata={"help": "Path to the test data. is list"})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps")
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=1)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")
    metric_for_best_model : str = field(default="mcrmse")
    greater_is_better: bool = field(default=False)
    stage: str = field(default='0')
    model_type: str = field(default='rna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")
    attn_implementation: str = field(default="eager")
    dataloader_num_workers: int = field(default=4)
    dataloader_prefetch_factor: int = field(default=2)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


"""
Transform a sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from    sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each    sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int, is_test_set=None) -> List[str]:
    """Load or generate k-mer string for each    sequence."""
    if is_test_set == 'public' or is_test_set == 'private':
        kmer_path = data_path.replace(".json", f"{is_test_set}_{k}mer.json")
    else:
        kmer_path = data_path.replace(".json", f"_{k}mer.json")
    print(kmer_path)
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer

def bpe_position(texts,attn_mask, tokenizer):
    position_id = torch.zeros(attn_mask.shape)
    for i,text in enumerate(texts):   
        text = tokenizer.tokenize(text)
        position_id[:, 0] = 1 #[cls]
        index = 0
        for j, token in enumerate(text):
            index = j+1
            position_id[i,index] = len(token) #start after [cls]   
        position_id[i, index+1] = 1 #[sep]
        
    print(position_id[0,:])
    print('position_id.shape',position_id.shape)
    return position_id

class SupervisedDataset(Dataset):

    def __init__(self, data_path, tokenizer,signal_noise_cutoff, test_set=None, kmer=-1,args=None):
        super().__init__()
        self.df = pd.read_json(data_path)
        print('pre',self.df.shape)
        deg_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']
        
        self.is_test = test_set is not None or deg_cols[0] not in self.df.columns
        if self.is_test:
            self.df = self.df.query(("seq_length == 107" if test_set == 'public' else "seq_length == 130"))
            self.y = None
        else:
            self.df = self.df[self.df.signal_to_noise >= signal_noise_cutoff]
            self.y = np.stack([np.stack(self.df[col].values) for col in deg_cols], axis=-1)
        print('post',self.df.shape)
        self.sample_ids = self.df['id'].values
        texts = [d.upper().replace("U", "T") for d in self.df['sequence']]
               
        seq_length = len(texts[0])
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer, test_set)

            if torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        # ensure tokenier
        print(type(texts[0]))
        print(texts[0])
        test_example = tokenizer.tokenize(texts[0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(texts[0]))
        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        self.input_ids = output["input_ids"]
        self.texts =texts
        #make sure the length of sequences in the dataset is the same
        self.weight_mask = torch.ones((self.input_ids.shape[0],seq_length+2))

        self.attention_mask = output["attention_mask"]
        if 'mer' in args.token_type:
            for i in range(1,kmer-1):
                self.weight_mask[:,i+1]=self.weight_mask[:,-i-2]=1/(i+1) 
            self.weight_mask[:, kmer:-kmer] = 1/kmer
        self.post_token_length = torch.zeros(self.attention_mask.shape)
        if args.token_type == 'bpe' or args.token_type == 'non-overlap':
            self.post_token_length = bpe_position(self.texts,self.attention_mask,tokenizer)
        self.num_labels = 3
    def __getitem__(self, index: int):
        if self.is_test:          
            sample_id = self.sample_ids[index]
            return dict(input_ids=self.input_ids[index], sample_ids=sample_id, attention_mask=self.attention_mask[index],
                weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index])
        targets = torch.tensor(self.y[index, :, :], dtype=torch.float32)
        return dict(input_ids=self.input_ids[index], labels=targets, attention_mask=self.attention_mask[index],
           weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index])
     
    
    def __len__(self) -> int:
        return self.df.shape[0]
@dataclass
class TestDataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, sample_ids, attention_mask, weight_mask, post_token_length = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"sample_ids", "attention_mask","weight_mask","post_token_length"))
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        
        return dict(
            input_ids=input_ids,
            sample_ids=sample_ids,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length
        )
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask, weight_mask, post_token_length  = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length"))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length
        )

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    def rmse(labels,logits):
        return np.mean(np.square(labels - logits + 1e-6))
    score = 0
    num_scored = 3
    for i in range(num_scored):
        score += rmse(labels[:, :, i], logits[:, :, i]) / num_scored       
    return {
        "mcrmse": score
    }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)

def build_submission_df(ids, pred_tensor):
    if type(pred_tensor).__module__ != np.__name__:
        pred_tensor = pred_tensor.cpu().detach().numpy()
    res = []
    for i, id in enumerate(ids):
        
        for j, pred in enumerate(pred_tensor[i, :, :]):
            res.append([id+'_'+str(j)] + list(pred))
    return res

def make_pred_file(args, model, loaders, postfix=''):
    res = []
    model.to(args.device)
    print(args.device)
    model.eval()
    for eval_dataloader in loaders:
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch["attention_mask"].to(args.device)
            sample_ids = batch["sample_ids"]
            weight_mask = batch["weight_mask"].to(args.device)
            post_token_length = batch["post_token_length"].to(args.device)
            with torch.no_grad():
                test_pred = model(input_ids=input_ids, attention_mask=attention_mask,weight_mask=weight_mask, post_token_length=post_token_length)
                test_pred = test_pred[0][:, 1:-1,:] #exclude [cls] and [sep]
                res += build_submission_df(sample_ids, test_pred)

    pred_df = pd.DataFrame(res, columns=['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])
    pred_df['deg_pH10'] = 0
    pred_df['deg_50C'] = 0
    results_path = os.path.join(args.output_dir,"results", args.run_name)
    print(results_path)
    os.makedirs(results_path, exist_ok=True)
    results_path = os.path.join(results_path, 'submission_'+postfix+'.csv')
    pred_df.to_csv(results_path, index=False)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'rnalm':
        tokenizer = EsmTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    elif training_args.model_type in ['rna-fm','rnabert','rnamsm','splicebert-human510','splicebert-ms510','splicebert-ms1024','utrbert-3mer','utrbert-4mer','utrbert-5mer','utrbert-6mer','utr-lm-mrl','utr-lm-te-el']:
        tokenizer = OpenRnaLMTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer=int(training_args.token_type[0])

    train_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_train_path), tokenizer, signal_noise_cutoff=0.6, test_set=None, kmer=data_args.kmer, args=training_args)
    val_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_val_path), tokenizer, signal_noise_cutoff=1.0, test_set=None, kmer=data_args.kmer, args=training_args)
    public_test_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_test_path), tokenizer, signal_noise_cutoff=-99.0, test_set='public', kmer=data_args.kmer, args=training_args)
    private_test_dataset = SupervisedDataset(os.path.join(data_args.data_path, data_args.data_test_path), tokenizer, signal_noise_cutoff=-99.0, test_set='private', kmer=data_args.kmer, args=training_args)
    #print(len(public_test_dataset))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    test_data_collator = TestDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(private_test_dataset)}+{len(private_test_dataset)}')

    # load model
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            
            print('Train from scratch')
            config = RnaLmConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                
                )
            print(config)
            model =  RnaLmForNucleotideLevel(
                config,
                tokenizer=tokenizer,
                )
        else:
            print('Loading rnalm model')  
            print(train_dataset.num_labels)
            model =  RnaLmForNucleotideLevel.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                problem_type="regression",
                token_type=training_args.token_type,
                attn_implementation=training_args.attn_implementation,
                )
            
    elif training_args.model_type == 'rna-fm':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaFmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
    elif training_args.model_type == 'rnabert':      
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
    elif training_args.model_type == 'rnamsm':
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = RnaMsmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )        
    elif 'splicebert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = SpliceBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )       
    elif 'utrbert' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrBertForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )  
    elif 'utr-lm' in training_args.model_type:
        print(training_args.model_type)
        print(f'Loading {training_args.model_type} model')
        model = UtrLmForNucleotideLevel.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            num_labels=train_dataset.num_labels,
            trust_remote_code=True,
            problem_type="regression",
            token_type=training_args.token_type,
            tokenizer=tokenizer,
        )     
       

    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
   
   
    def test_dataset_loader(dataset, args):
        return DataLoader(
            dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=test_data_collator,
            num_workers=4
        )
    test_data_loader1 = test_dataset_loader(public_test_dataset,training_args)
    test_data_loader2 = test_dataset_loader(private_test_dataset,training_args)
    make_pred_file(training_args, model, [test_data_loader1, test_data_loader2],postfix=training_args.output_dir.split('/')[-1])
if __name__ == "__main__":
    train()

#how to get score:
#submit to kaggle with command like
#kaggle competitions submit -c stanford-covid-vaccine -f xxx/submission_yyy.csv -m "Message"
