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
#from dnabert2_source.bert_layers import BertForSequenceReg512ConcatStep2 as BertForSequenceClassificationReg
import random
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
import transformers
import sklearn
import scipy
import numpy as np
import re
from torch.utils.data import Dataset

# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
# )
from model.modeling_rnalm import BertForRNAMRL
from model.rnalm_config import RNALMConfig
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
    save_total_limit: int = field(default=2)
    #lr_scheduler_type: str = field(default="cosine_with_restarts")
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)
    report_to: str = field(default="tensorboard")
    metric_for_best_model : str = field(default="r^2")
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
Load or generate k-mer string for each sequence. The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
"""
def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """Load or generate k-mer string for each sequence."""
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
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]

        if len(data[0]) == 2:
            texts = [d[0].upper().replace("U", "T") for d in data]          
            labels = [float(d[1]) for d in data]
            
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        labels = np.array(labels)
        labels = labels.tolist()
        
        if kmer != -1:
            # only write file on the first process
            if torch.distributed.get_rank() not in [0, -1]:
                torch.distributed.barrier()

            logging.warning(f"Using {kmer}-mer as input...")
            texts = load_or_generate_kmer(data_path, texts, kmer)
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
        self.attention_mask = output["attention_mask"]

        self.labels = labels
        self.num_labels = 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor,]:
        return dict(input_ids=self.input_ids[i],labels=self.labels[i], attention_mask=self.attention_mask[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple([instance[key] for instance in instances] for key in ("input_ids" ,"labels", "attention_mask"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = torch.Tensor(labels).float()      
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=padded_attention_mask,
        )

"""
Manually calculate the mse and r^2.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze()
    logits = logits.squeeze()
    #print(logits.shape,labels.shape)
    return {
    "mse": sklearn.metrics.mean_squared_error(labels, logits),
    "r^2" : scipy.stats.pearsonr(labels, logits)[0]**2,
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return calculate_metric_with_sklearn(logits, labels)



def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    set_seed(training_args)
    # load tokenizer
    if training_args.model_type == 'hyena':
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


    # load model
    # from DNA_BERT2_model.bert_layers import BertForSequenceClassification
    if training_args.model_type == 'rnalm':
        if training_args.train_from_scratch:
            #print('Loading 6mer model')
            print('Train from scratch')
            config = RNALMConfig.from_pretrained(model_args.model_name_or_path,
                num_labels=train_dataset.num_labels,
                problem_type="regression",
                token_type=training_args.token_type,
                use_flash_att = False,
                )
            print(config)
            model =  BertForRNAMRL(
                config
                )
        else:
            print('Loading rnabert model')
            #config = MMoeBertConfig.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir)
            #config.use_flash_attn = False
            print(train_dataset.num_labels)
            #config.num_labels=train_dataset.num_labels
            #from transformers import BertForSequenceClassification
            model =  BertForRNAMRL.from_pretrained(
                model_args.model_name_or_path,
                #config = config,
                cache_dir=training_args.cache_dir,
                num_labels=train_dataset.num_labels,
                #trust_remote_code=True,
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

    # define trainer
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
        #safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    # get the evaluation results from trainer
    if training_args.eval_and_save_results:
        results_path = os.path.join(training_args.output_dir, "results", training_args.run_name)
        
        os.makedirs(results_path, exist_ok=True)
        results_test = trainer.evaluate(eval_dataset=test_dataset)
        with open(os.path.join(results_path, "test_results.json"), "w") as f:
            json.dump(results_test, f)
        results_val = trainer.evaluate(eval_dataset=val_dataset)
        with open(os.path.join(results_path, "val_results.json"), "w") as f:
            json.dump(results_val, f)


if __name__ == "__main__":
    train()