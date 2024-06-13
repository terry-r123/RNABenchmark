import warnings
warnings.filterwarnings("ignore")
import os
import wandb
import numpy as np
import time
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import get_cosine_schedule_with_warmup

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs

from structure.data import SSDataset, ContactMapDataset
from structure.lm import get_extractor
from structure.predictor import SSCNNPredictor
import scipy
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import json
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(args.seed)
    print(f"seed is fixed ,seed = {args.seed}")

def bpe_position(texts,attn_mask, tokenizer):
    position_id = torch.zeros(attn_mask.shape)
    for i,text in enumerate(texts):   
        text = tokenizer.tokenize(text)
        position_id[:, 0] = 1
        index = 0
        for j, token in enumerate(text):
            index = j+1
            position_id[i,index] = len(token) #start after [cls]   
        position_id[i, index+1] = 1
    return position_id

def calculate_metric_with_sklearn(logits_list: [np.ndarray], labels_list: [np.ndarray]):
    # we calculate top-l, top-l/2, top-l/5 , top-l/10 long-range precision
    # long-range means the sequential distance between two residues is larger or equal to 24
    # precision = TP / (TP + FP)
    top_L_1_TP = []
    top_L_1_FP = []
    top_L_2_TP = []
    top_L_2_FP = []
    top_L_5_TP = []
    top_L_5_FP = []
    top_L_10_TP = []
    top_L_10_FP = []
    lengths = np.array([labels.shape[-1] for labels in labels_list])
    long_range_mask = np.zeros((lengths.max(), lengths.max()))
    long_range_mask[np.triu_indices(long_range_mask.shape[0], k=23)] = 1
    # make it boolean
    long_range_mask = long_range_mask.astype(bool)
    for logits, labels in zip(logits_list, labels_list):
        labels = labels.squeeze().astype(float)
        logits = logits.squeeze()
        logits = (logits + logits.T) / 2 # symmetrize the logits, we only consider the upper triangle
        predictions = scipy.special.expit(logits)
        long_range_mask_tmp = long_range_mask[:labels.shape[-1], :labels.shape[-1]]
        # long range only
        long_range_labels = labels[long_range_mask_tmp].flatten()
        long_range_predictions = predictions[long_range_mask_tmp].flatten()

        L = labels.shape[-1]
        for factor in [1,2,5,10]:
            length = L // factor
            # get indices of the top Length predictions
            top_L_indices = np.argsort(long_range_predictions)[-length:]
            top_L_predictions = long_range_predictions[top_L_indices]
            top_L_labels = long_range_labels[top_L_indices]
            top_L_predictions_over_threshold = top_L_predictions > 0.5

            true_positives = top_L_labels[top_L_predictions_over_threshold].sum()
            false_positives = (1 -top_L_labels[top_L_predictions_over_threshold]).sum()

            # append to the list
            if factor == 1:
                top_L_1_TP.append(true_positives)
                top_L_1_FP.append(false_positives)
            elif factor == 2:
                top_L_2_TP.append(true_positives)
                top_L_2_FP.append(false_positives)
            elif factor == 5:
                top_L_5_TP.append(true_positives)
                top_L_5_FP.append(false_positives)
            elif factor == 10:
                top_L_10_TP.append(true_positives)
                top_L_10_FP.append(false_positives)
            
    # calculate the precision
    top_L_1_precision = sum(top_L_1_TP) / (sum(top_L_1_TP) + sum(top_L_1_FP))
    top_L_2_precision = sum(top_L_2_TP) / (sum(top_L_2_TP) + sum(top_L_2_FP))
    top_L_5_precision = sum(top_L_5_TP) / (sum(top_L_5_TP) + sum(top_L_5_FP))
    top_L_10_precision = sum(top_L_10_TP) / (sum(top_L_10_TP) + sum(top_L_10_FP))

    return {
        "top_l_precision": top_L_1_precision,
        "top_l/2_precision": top_L_2_precision,
        "top_l/5_precision": top_L_5_precision,
        "top_l/10_precision": top_L_10_precision,
    }

class collator():
    def __init__(self,tokenizer,args):
        self.tokenizer = tokenizer
        self.args = args
    def __call__(self,batch):
        seqs= [x['seq'] for x in batch]
        struct = [x['struct'] for x in batch]
     
        max_len = max([len(seq) for seq in seqs])
        #max_len = min(max_len, self.tokenizer.model_max_length)
        
        weight_mask = torch.ones((len(seqs), max_len+2)) #including [cls] and [sep], dim= [bz, max_len+2]
        if 'mer' in self.args.token_type:
            kmer=int(self.args.token_type[0])
            for i in range(1,kmer-1):
                weight_mask[:,i+1]=weight_mask[:,-i-2]=1/(i+1) 
            weight_mask[:, kmer:-kmer] = 1/kmer
            seqs = [generate_kmer_str(seq, kmer) for seq in seqs]

        data_dict = self.tokenizer(seqs, 
                        padding='longest', 
                        max_length=self.tokenizer.model_max_length, 
                        truncation=True, 
                        return_tensors='pt')
        post_token_length = torch.zeros(data_dict['attention_mask'].shape)
        if self.args.token_type == 'bpe' or args.token_type == 'non-overlap':
            post_token_length = bpe_position(seqs,data_dict['attention_mask'],self.tokenizer)
        
        input_ids = data_dict["input_ids"]  

        # each elemenet in struct is a square matrix, but with different shape. We need to pad them to make them the same shape
        struct =  np.array( [np.pad(x, ((0,max_len-x.shape[0]),(0,max_len-x.shape[1])), 'constant', constant_values=-1) for x in struct])
        struct = torch.tensor(struct)

        data_dict['struct'] = struct
        data_dict['weight_mask'] = weight_mask
        data_dict['post_token_length'] = post_token_length

        return data_dict

def test(model, test_loader, accelerator):
    model.eval()  # Set the model to evaluation mode
    outputs_list = []
    targets_list = []
    with torch.no_grad(): 
        for data_dict in tqdm(test_loader):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(accelerator.device)

            logits = model(data_dict)[:,1:-1,1:-1]
            labels = data_dict['struct']
            label_mask = labels != -1
            outputs_list.append(logits.detach().cpu().numpy())
            targets_list.append(labels.detach().cpu().numpy())
            
    metrics = calculate_metric_with_sklearn(outputs_list, targets_list)
    print(f'\nTest: Top-l precision: {metrics["top_l_precision"]}, Top-l/2 precision: {metrics["top_l/2_precision"]}, Top-l/5 precision: {metrics["top_l/5_precision"]}, Top-l/10 precision: {metrics["top_l/10_precision"]}')
    return metrics

def main(args):
    set_seed(args)
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with='wandb')
    model_name = args.output_dir.split('/')[-1]

    name = f'[RNA_Secondary_Structure_Prediction]{model_name}_' \
           f'{args.model_type}_' \
           f'{args.token_type}_' \
           f'lr{args.lr}_' \
           f'bs{args.per_device_train_batch_size}*gs{args.gradient_accumulation_steps}*gpu{accelerator.state.num_processes}_' \
           f'gs{args.gradient_accumulation_steps}' \
           f'epo{args.num_epochs}_' \
           f'warm{args.warmup_epos}epoch' \
           f'seed{args.seed}'


    if args.is_freeze:
        name += '_freeze'
 
    model_config = extractor.config
    model = SSCNNPredictor(args, extractor, model_config, tokenizer, args.is_freeze)
    num_params = count_parameters(model)

    print(f"model params are: {num_params}")

    
    train_dataset = ContactMapDataset(data_path=os.path.join(args.data_path, args.data_train_path), tokenizer=tokenizer, args=args)
    val_dataset = ContactMapDataset(data_path=os.path.join(args.data_path, args.data_val_path), tokenizer=tokenizer, args=args)
    test_dataset_list = []
    data_test_list = args.data_test_path.replace(" ", "").split(",")
    for data_test in data_test_list:
        data_test_name = data_test +".csv"
        print(f"evaluating data_test_name = {data_test_name}")
        test_dataset = ContactMapDataset(data_path=os.path.join(args.data_path, data_test_name), tokenizer=tokenizer, args=args)
        test_dataset_list.append(test_dataset)
    
    
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset_list[0])}+{len(test_dataset_list[1])}+{len(test_dataset_list[2])}')
    collate_fn = collator(tokenizer,args)
    train_loader = DataLoader(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    test_dataloader_list = []
    for test_dataset in test_dataset_list:
        test_loader = DataLoader(test_dataset, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)
        test_dataloader_list.append(test_loader)                     

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## num_processes from accelerator
    per_steps_one_epoch = len(
        train_dataset) // args.per_device_train_batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    num_warmup_steps = per_steps_one_epoch * args.warmup_epos
    num_training_steps = per_steps_one_epoch * args.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    if accelerator.is_main_process:
        wandb.init(project='ContactMap', mode='offline')
        wandb.run.name = name
        wandb.run.save()
        wandb.watch(model)
        print(name)

    criterion = nn.BCEWithLogitsLoss()

    train_loss_batch_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    step = 0
    last_val, best_val, best_test = 0, 0, []
    patience = args.patience
    early_stop_flag = 0
    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        for data_dict in tqdm(train_loader):
            with accelerator.accumulate(model):
                logits = model(data_dict)[:,1:-1,1:-1]               
                labels = data_dict['struct']
                label_mask = labels != -1 
                loss = criterion(logits[label_mask].reshape(-1,1), labels[label_mask].reshape(-1,1))
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            

            if accelerator.sync_gradients:
                lr_scheduler.step()

            gather_loss = accelerator.gather(loss.detach().float()).mean().item()
            train_loss_list.append(gather_loss)
     
        val_auc_list, val_recall_list, val_precision_list, val_f1_list = [], [], [], []
        test_auc_list, test_recall_list, test_precision_list, test_f1_list = [], [], [], []

        threshold = 0.5
        print(f"epoch {epoch}:")
        val_metrics = test(model, val_loader, accelerator)

        if best_val < val_metrics["top_l_precision"]:
            best_val = val_metrics["top_l_precision"] 
            print(f"epoch {epoch}:")
            test_metrics = []   
            for i,data_test in enumerate(data_test_list):
                test_metrics.append(test(model, test_dataloader_list[i], accelerator))
            best_test = test_metrics
        if last_val < val_metrics["top_l_precision"]:
            early_stop_flag = 0 
        else:
            early_stop_flag +=1

        if early_stop_flag >= patience:
            print(f"Early stopping")
            break
        last_val = val_metrics["top_l_precision"]

        end_time = time.time()

        if accelerator.is_main_process:
            print(
                f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.2f}')
            log_dict = {'lr': optimizer.param_groups[0]["lr"], 'train_loss': np.mean(train_loss_list)}
            log_dict.update(val_metrics)
            for test_metrics in best_test:
                log_dict.update(test_metrics)
            wandb.log(log_dict)
        torch.cuda.empty_cache()
        train_loss_list, val_loss_list, test_loss_list = [], [], []

    
    for i,data_test in enumerate(data_test_list):
        results_path = os.path.join(args.output_dir, "results", args.run_name)
        os.makedirs(results_path, exist_ok=True)
        with open(os.path.join(results_path, f"{data_test}_results.json"), "w") as f:
            json.dump(best_test[i], f, indent=4)
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epos', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--model_scale', type=str, default='8m')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='bprna')
    parser.add_argument("--pretrained_lm_dir", type=str, default='')
    parser.add_argument('--data_path', default= '')
    parser.add_argument('--model_name_or_path', default='output')
    parser.add_argument('--output_dir', default='./ckpts/')
    parser.add_argument('--model_type', type=str, default='rna')
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--bprna_dir', default='')
    parser.add_argument('--run_name', type=str, default="run")
    parser.add_argument('--token_type', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--train_from_scratch', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_train_path', default= '')
    parser.add_argument('--data_val_path', default= '')
    parser.add_argument('--data_test_path', default= '')
    parser.add_argument('--attn_implementation', type=str, default="eager")
    args = parser.parse_args()

    assert args.mode in ['bprna', 'pdb']

    ## get pretrained model
    extractor, tokenizer = get_extractor(args)

    main(args)
