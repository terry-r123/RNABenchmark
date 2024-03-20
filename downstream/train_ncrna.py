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


os.environ["WANDB_DISABLED"] = "true"


#import IA3_lora
from transformers import Trainer, TrainingArguments, BertTokenizer,EsmTokenizer, EsmModel, AutoConfig, AutoModel, EarlyStoppingCallback
# import sys
# sys.path.append("..") 

from model.rnalm.modeling_rnalm import BertForSequenceClassification
from model.rnalm.rnalm_config import RNALMConfig
early_stopping = EarlyStoppingCallback(early_stopping_patience=20)
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
    metric_for_best_model: str = field(default="accuracy")
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
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0].upper().replace("U", "T") for d in data]
            labels = [int(d[1]) for d in data]
        else:
            print(len(data[0]))
            raise ValueError("Data format not supported.")
        
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
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        #print('1',input_ids[0].shape)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        #print('2',input_ids[0].shape)
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": sklearn.metrics.accuracy_score(labels, predictions),
        "f1": sklearn.metrics.f1_score(labels, predictions, average="macro", zero_division=0),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(labels, predictions),
        "precision": sklearn.metrics.precision_score(labels, predictions, average="macro", zero_division=0),
        "recall": sklearn.metrics.recall_score(labels, predictions, average="macro", zero_division=0),
    }

"""
Compute metrics used for huggingface trainer.
"""
def compute_metrics(eval_pred):
    logits, labels = eval_pred
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
    train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    val_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "val.csv"), 
                                     kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


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
            print('Loading esm model')
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
        #model.print_trainable_parameters()

    # if model_args.apply_ia3:
    #     peft_config = IA3_lora.LoRAConfig(lora_rank=model_args.lora_r, lora_init_scale=model_args.lora_init_scale)
    #     model = IA3_lora.modify_with_lora(model, peft_config)
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
                                   data_collator=data_collator,
                                   callbacks=[early_stopping],
                                   )
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

