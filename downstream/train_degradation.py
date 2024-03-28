import os
import csv
import copy
import json
import logging
import pdb
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
# from model_regression_bert_flash import BertForSequenceClassification as BertForSequenceClassification_flash
# from model_regression_bert_flash_concat import BertForSequenceClassification as BertForSequenceClassification_flash
# from model_regression_nt import EsmForSequenceClassification
#from dnabert2_source.bert_layers import BertForSequenceRNAdegra as DNABERT2ForRNAdegra
import random

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

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


from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
from model.rnalm.modeling_rnalm import BertForSequenceRNAdegra 
from model.rnalm.rnalm_config import RNALMConfig
#from mmoe.modeling_bert import BertForSequenceClassification
#from mmoe.modeling_bert_train import BertForSequenceRNAdegra 
#from mmoe.modeling_esm import ESMForSequenceRNAdegra 
#from hyena_dna.standalone_hyenadna import HyenaForRNADegraPre, CharacterTokenizer
from model.esm.modeling_esm import ESMForSequenceRNAdegra
from model.esm.esm_config import EsmConfig
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


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    delete_n: bool = field(default=False, metadata={"help": "data delete N"})


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
    save_total_limit: int = field(default=2)
    #lr_scheduler_type: str = field(default="cosine_with_restarts")
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)
    report_to: str = field(default="wandb")
    metric_for_best_model : str = field(default="mcrmse")
    greater_is_better: bool = field(default=False)
    stage: str = field(default='0')
    model_type: str = field(default='dna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")




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

def remove_non_acgt_chars(sequence):
    pattern = '[^ACGT]'
    cleaned_sequence = re.sub(pattern, '', sequence)
    return cleaned_sequence

def replace_consecutive_ns(sequence, n=10):
    pattern = 'N' * n + '+'
    return re.sub(pattern, 'N', sequence)


"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

def count_bases(sequence):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0, 'Others': 0}
    total_chars = len(sequence)
    others_count = 0
    max_percentage = 0
    for char in sequence:
        if char in counts:
            counts[char] += 1
        else:
            counts['Others'] += 1
    for char, count in counts.items():
        percentage = (count / total_chars) * 100
        if percentage > 0 and char == 'Others':
            # pdb.set_trace()
            max_percentage = max(percentage, max_percentage)
            print(f"{char}: {percentage:.2f}%, sequence = {sequence}")
            others_count += 1
    return others_count, max_percentage

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int, is_test_set=None) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
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
        #print(text)
        text = tokenizer.tokenize(text)
        #print(text)
        for j, token in enumerate(text):
            position_id[i,j] = len(token)    
        #print(tokenizer)  
    print(position_id)
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
        self.X = np.stack(self.df['train_tensor'].values)
        #texts = self.df['sequence'].values
        texts = [d.upper().replace("U", "T") for d in self.df['sequence']]
               
        #self.id_to_bp_mat_map = self.load_bp_mats()
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
        #print('------------------',self.weight_mask.shape)

        self.attention_mask = output["attention_mask"]
        #print(self.attention_mask[0])
        if args.token_type == '6mer':
            for i in range(1,5):
                self.weight_mask[:,i+1]=self.weight_mask[:,-i-2]=1/(i+1) 
            self.weight_mask[:, 6:-6] = 1/6
        self.post_token_length = torch.zeros(self.attention_mask.shape)
        if args.token_type == 'bpe':
            self.post_token_length = bpe_position(self.texts,self.attention_mask,tokenizer)
        #self.weight_mask = torch.cat([torch.ones((self.input_ids.shape[0],1)),self.weight_mask,torch.ones((self.input_ids.shape[0],1))],dim=1) #add [cls] [sep]
        #print(self.weight_mask)
        #print(self.position_id)
        self.num_labels = 3
    def __getitem__(self, index: int):
        if self.is_test:
            #print(self.sample_ids.shape,self.input_ids.shape,self.attention_mask.shape)
            sample_id = self.sample_ids[index]
            return dict(input_ids=self.input_ids[index], sample_ids=sample_id, attention_mask=self.attention_mask[index],
                weight_mask=self.weight_mask[index],post_token_length=self.post_token_length[index])
        #print('self.texts[index]',self.texts[index])
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
        input_ids, labels, attention_mask, weight_mask, post_token_length  = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length"))
        input_ids = torch.stack(input_ids)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        #attention_mask = torch.cat([attention_mask,weight_mask,post_token_length],dim=1)
        
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
        #print(instances)
        input_ids, labels, attention_mask, weight_mask, post_token_length  = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask","weight_mask","post_token_length"))
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        attention_mask = torch.stack(attention_mask)
        weight_mask = torch.stack(weight_mask)
        post_token_length = torch.stack(post_token_length)
        #print(input_ids.shape,post_token_length.shape)
        #print(weight_mask.shape,attention_mask.shape,labels.shape)
        #attention_mask = torch.cat([attention_mask,weight_mask,post_token_length],dim=1)
        #print(weight_mask.shape,attention_mask.shape,labels.shape)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            weight_mask=weight_mask,
            post_token_length=post_token_length
        )

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    def rmse(labels,logits):
        #print('rmse',labels.shape,logits.shape)
        return np.mean(np.square(labels - logits + 1e-6))
    #logits = logits[:, 1:1+labels.shape[1], :]
    #print(logits.shape)
    #print(logits.shape[2])     
    score = 0
    num_scored = 3
    #print('mcrmse',labels.shape,logits.shape)
    for i in range(num_scored):
        score += rmse(labels[:, :, i], logits[:, :, i]) / num_scored       
    return {
        "mcrmse": score
    }
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    #print(logits.shape,labels.shape)
    return calculate_metric_with_sklearn(logits, labels)

def build_submission_df(ids, pred_tensor):
    if type(pred_tensor).__module__ != np.__name__:
        pred_tensor = pred_tensor.cpu().detach().numpy()
    res = []
    #print(ids)
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
            weight_mask = batch["weight_mask"]
            post_token_length = batch["post_token_length"]
            with torch.no_grad():
                test_pred = model(input_ids=input_ids, attention_mask=attention_mask,weight_mask=weight_mask, post_token_length=post_token_length)
                # print(len(outputs))
                #print(test_pred[0].shape)
                test_pred = test_pred[0][:, 1:-1,:] #exclude [cls] and [sep]
                #print(test_pred.shape)
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
    if training_args.model_type == 'esm-rna':
        tokenizer = EsmTokenizer.from_pretrained("/mnt/data/ai4bio/renyuchen/DNABERT/examples/rna_finetune/ssp/vocab_esm_mars.txt")
    elif training_args.model_type == 'hyena':
        tokenizer = CharacterTokenizer(
            characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
            model_max_length=training_args.model_max_length + 2,  # to account for special tokens, like EOS
            add_special_tokens=False,  # we handle special tokens elsewhere
            padding_side='left', # since HyenaDNA is causal, we pad on the left
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
    # prepd_train_data = '/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test/train_1.json'
    # prepd_val_data = '/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test/val_1.json'
    # test_data = '/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test/test_1.json'
    train_dataset = SupervisedDataset(os.path.join(data_args.data_path,'train_1.json'), tokenizer, signal_noise_cutoff=0.6, test_set=None, kmer=data_args.kmer, args=training_args)
    val_dataset = SupervisedDataset(os.path.join(data_args.data_path,'val_1.json'), tokenizer, signal_noise_cutoff=1.0, test_set=None, kmer=data_args.kmer, args=training_args)
    public_test_dataset = SupervisedDataset(os.path.join(data_args.data_path,'test_1.json'), tokenizer, signal_noise_cutoff=-99.0, test_set='public', kmer=data_args.kmer, args=training_args)
    private_test_dataset = SupervisedDataset(os.path.join(data_args.data_path,'test_1.json'), tokenizer, signal_noise_cutoff=-99.0, test_set='private', kmer=data_args.kmer, args=training_args)
    #print(len(public_test_dataset))
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    test_data_collator = TestDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(private_test_dataset)}+{len(private_test_dataset)}')

    # load model
    # from DNA_BERT2_model.bert_layers import BertForSequenceClassification
    if training_args.model_type=='mmoe':
        config = MMoeBertConfig.from_pretrained(model_args.model_name_or_path,
        num_labels=train_dataset.num_labels)
        config.stage = training_args.stage
        print(config)
        model = MMoeBertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config = config,
        cache_dir=training_args.cache_dir,
        #num_labels = train_dataset.num_labels,
        )
        if config.stage == '1':
            for name, param in model.named_parameters():
                for key in ["rna_expert", "rna_embeddings","rna_pooler", "rna_LayerNorm"]:
                    if key in name:
                        param.requires_grad = False  
                        break        
        elif config.stage == '2':
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                for key in ["rna_expert", "rna_embeddings","rna_pooler", "rna_LayerNorm","classifier","attention"]:
                    if key in name:
                        #pdb.set_trace()
                        param.requires_grad = True
   
    elif training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            
            print('Train from scratch')
            config = RNALMConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                use_flash_att = False,
                )
            print(config)
            model =  BertForSequenceRNAdegra(
                config
                )
        else:
            print('Loading rnalm model')
            #config = MMoeBertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
            #config.use_flash_attn = False
            print(train_dataset.num_labels)
            #config.num_labels=train_dataset.num_labels
            #from transformers import BertForSequenceClassification
            model =  BertForSequenceRNAdegra.from_pretrained(
                model_args.model_name_or_path,
                #config = config,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                #trust_remote_code=True,
                problem_type="regression",
                token_type=training_args.token_type,
                )
            
    elif training_args.model_type == 'rna-fm' or training_args.model_type == 'esm-rna':
        if training_args.train_from_scratch:
            print(f'Loading {training_args.model_type} model')
            print('Train from scratch')
            config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels)
            model = ESMForSequenceRNAdegra(
                config
                )
        else:
            print(training_args.model_type)
            print(f'Loading {training_args.model_type} model')
            model = ESMForSequenceRNAdegra.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                trust_remote_code=True,
            )        
   
    elif training_args.model_type == 'dnabert2':
        if training_args.train_from_scratch:
            pass
        else:
            print('Loading dnabert2 model')          
            print(train_dataset.num_labels)
            model = DNABERT2ForRNAdegra.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                use_alibi=model_args.use_alibi,
                problem_type="regression",
                token_type=training_args.token_type,
            )
    elif training_args.model_type == 'hyena':
        backbone_cfg = None
        if training_args.train_from_scratch:
            pass
        else:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print("Using device:", device)
            model = HyenaForRNADegraPre.from_pretrained(
                model_args.model_name_or_path,
                #download=True,
                config=backbone_cfg,
                device=device,
                use_head=False,
                n_classes=train_dataset.num_labels,
                problem_type="regression",
                #token_type=training_args.token_type,
            )

    # configure LoRA
    if model_args.use_lora:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=list(model_args.lora_target_modules.split(",")),
            lora_dropout=model_args.lora_dropout,
            bias="none",
            task_type="SEQ_CLS",
            inference_mode=False,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
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
   
    # test_data_path = '/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test/test_1.json'
    # test_data_loader1 = dataset_loader(test_data_path, test_set='public', batch_size=batch_size)
    # test_data_loader2 = dataset_loader(test_data_path, test_set='private', batch_size=batch_size)
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
#kaggle competitions submit -c stanford-covid-vaccine -f /mnt/data/oss_beijing/renyuchen/temp/ft/rna-all/degra/dna/open/dnabert1/results/dnabert_seed42/submission_dnabert1.csv -m "Message"
