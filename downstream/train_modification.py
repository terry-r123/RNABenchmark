import os
import csv
import copy
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List

import torch
import random
import transformers
import sklearn
import numpy as np
from torch.utils.data import Dataset
import pdb

import pandas as pd
os.environ["WANDB_DISABLED"] = "true"
from sklearn.metrics import roc_auc_score, matthews_corrcoef 


from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel

from model.rnalm.modeling_rnalm import BertForSequenceClassification
from model.rnalm.rnalm_config import RNALMConfig

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    apply_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    apply_ia3: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_init_scale: float = field(default=0.01, metadata={"help": "dropout rate for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})
    use_alibi: bool = field(default=True, metadata={"help": "whether to use alibi"})

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


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
    evaluation_strategy: str = field(default="steps"),
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
    fp16: bool = field(default=False)
    report_to: str = field(default="tensorboard")
    metric_for_best_model: str = field(default="mean_mcc")
    stage: str = field(default='0')
    model_type: str = field(default='dna')
    token_type: str = field(default='6mer')
    train_from_scratch: bool = field(default=False)
    log_dir: str = field(default="output")

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.distributed.get_rank() >= 0:
        print("!!!!!!!!!!!!!", "Yes")
        torch.cuda.manual_seed_all(args.seed)

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])

"""
Transform a dna sequence to k-mer string
"""
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


"""
Load or generate k-mer string for each DNA sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each DNA sequence."""
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
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

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, data_args,
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        self.data = pd.read_csv(data_path, sep=",", header=None, dtype={i: np.int8 for i in range(1, 920)})
        self.data["targets"] = list(self.data.iloc[:, 1:].values)
        self.data = self.data[[0, "targets"]]
        self.data.columns = ["seq", "targets"]
        self.num_labels = 12
        self.kmer = kmer
        self.tokenizer = tokenizer
         # ensure tokenier
        print(type(self.data["seq"][0]))
        print(self.data["seq"][0])
        test_example = tokenizer.tokenize(self.data["seq"][0])
        print(test_example)
        print(len(test_example))
        print(tokenizer(self.data["seq"][0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.data["seq"][idx].upper()
        labels = self.data["targets"][idx].astype(np.float32)
        if self.kmer != -1:
            sample = generate_kmer_str(sample,self.kmer)
        #print(sample)

        output = self.tokenizer(
            sample, 
            return_tensors="pt",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,)

        input_ids = output["input_ids"][0]
        #print(input_ids)
        attention_mask = output["attention_mask"][0]
        # features["labels"] = targets.astype(np.float32)
        return dict(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "attention_mask"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # labels = torch.Tensor(labels).float()
        labels = torch.tensor(np.array(labels)).float()
        # pdb.set_trace()
        attention_mask = torch.stack(attention_mask)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    metrics = {}
    # logits_torch = torch.from_numpy(logits).float()
    # p = torch.sigmoid(logits_torch)
    p = torch.sigmoid(logits)
    y = labels
    # y, p = data['labels'], torch.sigmoid(data['predictions'])
    # compute auc for each class independetly, https://github.com/jimmyyhwu/deepsea/blob/master/compute_aucs.py#L46
    aucs = np.zeros(12, dtype=np.float32)
    mcc_scores = []
    for i in range(12):
        try:
            mcc_score = matthews_corrcoef(y[:, i], p[:, i] > 0.5)
            mcc_scores.append(mcc_score)
            aucs[i] = roc_auc_score(y[:, i], p[:, i])
        except ValueError:
            aucs[i] = 0.5
    metrics['hAm_auc'] = float(np.median(aucs[0]))
    metrics['hCm_auc'] = float(np.median(aucs[1]))
    metrics['hGm_auc'] = float(np.median(aucs[2]))
    metrics['hUm_auc'] = float(np.median(aucs[3]))
    metrics['hm1A_auc'] = float(np.median(aucs[4]))
    metrics['hm5C_auc'] = float(np.median(aucs[5]))
    metrics['hm5U_auc'] = float(np.median(aucs[6]))
    metrics['hm6A_auc'] = float(np.median(aucs[7]))
    metrics['hm6Am_auc'] = float(np.median(aucs[8]))
    metrics['hm7G_auc'] = float(np.median(aucs[9]))
    metrics['hPsi_auc'] = float(np.median(aucs[10]))
    metrics['Atol_auc'] = float(np.median(aucs[11]))
    #metrics['median_auc'] = float(np.median(metrics['hAm']+metrics['hCm']+metrics['hGm']+metrics['hUm']+metrics['hm1A']+metrics['hm5C']+metrics['hm5U']+metrics['hm6A']+metrics['hm6Am']+metrics['hm7G']+metrics['hPsi']+metrics['Atol'])/12.0)
    metrics['mean_auc'] = (metrics['hAm_auc']+metrics['hCm_auc']+metrics['hGm_auc']+metrics['hUm_auc']+metrics['hm1A_auc']+metrics['hm5C_auc']+metrics['hm5U_auc']+metrics['hm6A_auc']+metrics['hm6Am_auc']+metrics['hm7G_auc']+metrics['hPsi_auc']+metrics['Atol_auc']) / 12.0
    metrics['mean_mcc'] = np.mean(mcc_scores)
    
    return metrics



"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'esm-open':
        tokenizer = EsmTokenizer.from_pretrained("/mnt/data/ai4bio/renyuchen/DNABERT/examples/rna_finetune/ssp/vocab_esm_mars.txt")
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            # trust_remote_code=True,
        )
    # token_test = "ATCGGCAGTACAGCGATTTGACGAT"
    # print(token_test)
    # print(tokenizer.tokenize(token_test))
    # print(tokenizer(token_test))
    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token
    if 'mer' in training_args.token_type:
        data_args.kmer=int(training_args.token_type[0])
    # define datasets and data collator
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args,
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args,
                                     data_path=os.path.join(data_args.data_path, "val.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, data_args=data_args,
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    # load model
    print(training_args.model_type)
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
    elif training_args.model_type == 'dnabert2':

        if training_args.train_from_scratch:
            pass
        else:
            print('Loading dnabert2 model')          
            print(train_dataset.num_labels)
            model = DNABERT2ForClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
                use_alibi=model_args.use_alibi,
                
            )
    elif training_args.model_type == 'rnalm':
            if training_args.train_from_scratch:
                
                print('Train from scratch')
                config = RNALMConfig.from_pretrained(model_args.model_name_or_path,
                    num_labels=train_dataset.num_labels)
                model = BertForSequenceClassification(
                    config
                    )
            else:
                print('Loading rnalm model')
                print(train_dataset.num_labels)

                model = BertForSequenceClassification.from_pretrained(
                    model_args.model_name_or_path,
                    #config = config,
                    cache_dir=training_args.cache_dir,
                    num_labels=train_dataset.num_labels,
                    #trust_remote_code=True,
                    )
            
    else:
        if training_args.train_from_scratch:
            print('Loading esm model')
            print('Train from scratch')
            config = AutoConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels)
            model = transformers.AutoModelForSequenceClassification.from_config(
                config
                )
        else:
            model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                trust_remote_code=True,
            )
    #print(model_args,training_args)
    # configure LoRA
    if model_args.apply_lora:
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

    if 'mmoe' in model_args.model_name_or_path:
        if torch.distributed.get_rank() in [0, -1]:
            for name, param in model.named_parameters():
                print(name, param.requires_grad)
            print(get_parameter_number(model))

    # define trainer
    trainer = transformers.Trainer(model=model,
                                   tokenizer=tokenizer,
                                   args=training_args,
                                   compute_metrics=compute_metrics,
                                   train_dataset=train_dataset,
                                   eval_dataset=val_dataset,
                                   data_collator=data_collator)
    trainer.train()

    if training_args.save_model:
        trainer.save_state()
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        results = trainer.evaluate(eval_dataset=test_dataset)
        print("on the test set:", results, "\n", results_path)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results, f)
         




if __name__ == "__main__":
    train()

