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

from structure.data import SSDataset, DistanceMapDataset
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
    # print(texts[0])
    # print(tokenizer.tokenize(texts[0]))
    for i,text in enumerate(texts):   
        text = tokenizer.tokenize(text)
        position_id[:, 0] = 1
        index = 0
        for j, token in enumerate(text):
            index = j+1
            position_id[i,index] = len(token) #start after [cls]   
            # if i == 0:
            #     print(token,position_id[i,index],i,index,len(token))
        position_id[i, index+1] = 1
        
    #print(position_id[0,:])
    #print('position_id.shape',position_id.shape)
    return position_id

def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    labels = labels.squeeze().astype(float)
    logits = logits.squeeze()
    #binary classification
    # Apply the sigmoid function to the logits to get the predicted probabilities
    R_2 = scipy.stats.pearsonr(labels, logits)[0] ** 2
    MSE = np.mean((labels - logits) ** 2)
    return {
        "r^2": R_2,
        "mse": MSE
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
        if self.args.token_type == '6mer':
            for i in range(1,5):
                weight_mask[:,i+1]=weight_mask[:,-i-2]=1/(i+1) 
            weight_mask[:, 6:-6] = 1/6
            seqs = [generate_kmer_str(seq, 6) for seq in seqs]

        data_dict = self.tokenizer(seqs, 
                        padding='longest', 
                        max_length=self.tokenizer.model_max_length, 
                        #add_special_tokens=False,
                        truncation=True, 
                        return_tensors='pt')
        post_token_length = torch.zeros(data_dict['attention_mask'].shape)
        if self.args.token_type == 'bpe' or args.token_type == 'non-overlap':
            post_token_length = bpe_position(seqs,data_dict['attention_mask'],self.tokenizer)
        
        input_ids = data_dict["input_ids"]  

        # each elemenet in struct is a square matrix, but with different shape. We need to pad them to make them the same shape
        #struct =  np.array( [np.pad(x, ((1,max_len-1-x.shape[0]),(1,max_len-1-x.shape[1])), 'constant', constant_values=-1) for x in struct])
        struct =  np.array( [np.pad(x, ((0,max_len-x.shape[0]),(0,max_len-x.shape[1])), 'constant', constant_values=-1) for x in struct])
        struct = torch.tensor(struct).float()

        data_dict['struct'] = struct
        data_dict['weight_mask'] = weight_mask
        data_dict['post_token_length'] = post_token_length

        return data_dict

def test(model, test_loader, accelerator):
    model.eval()  # Set the model to evaluation mode
    outputs_list = []
    targets_list = []
    #test_loss_list = []
    with torch.no_grad(): 
        for data_dict in tqdm(test_loader):
            for key in data_dict:
                data_dict[key] = data_dict[key].to(accelerator.device)

            logits = model(data_dict)[:,1:-1,1:-1]
            labels = data_dict['struct']
            #print(labels.shape)
            label_mask = labels != -1
            outputs_list.append(logits[label_mask].detach().cpu().numpy().reshape(-1,1))
            targets_list.append(labels[label_mask].detach().cpu().numpy().reshape(-1,1))
            #print(logits.shape,labels.shape)
            #loss = criterion(logits[label_mask].reshape(-1,1), labels[label_mask].reshape(-1,1))
            #test_loss_list.append(loss.item())
        logits = np.concatenate(outputs_list,axis = 0)
        labels = np.concatenate(targets_list,axis = 0)
    #print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    #print(len(outputs_list))
    metrics = calculate_metric_with_sklearn(logits, labels)
    #print('yyyyyyyyyyyyyyyyyyyyyyyyyyyy')
    print(f'\nTest R^2: {metrics["r^2"]}', f'   Test MSE: {metrics["mse"]}')
    return metrics

def main(args):
    set_seed(args)
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with='wandb')
    model_name = args.output_dir.split('/')[-1]
    #model_name = args.model_name_or_path.split('/')[-1]
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

    #args.pdb_dir = f'{args.data_dir}/RNA_Secondary_Structure_Prediction/PDB_SS'
    #args.bprna_dir = f'{args.data_dir}/bpRNA'

    # ckpt_path = os.path.join(args.output_dir, name)
    # os.makedirs(ckpt_path, exist_ok=True)

    
    

    model_config = extractor.config
    model = SSCNNPredictor(args, extractor, model_config, tokenizer, args.is_freeze)
    num_params = count_parameters(model)
    #print(model)
    print(f"model params are: {num_params}")
    #if args.mode == 'bprna':
        ## bprna data ##
    
    train_dataset = DistanceMapDataset(data_path=os.path.join(args.data_path, args.data_train_path), tokenizer=tokenizer, args=args)
    val_dataset = DistanceMapDataset(data_path=os.path.join(args.data_path, args.data_val_path), tokenizer=tokenizer, args=args)
    test_dataset_list = []
    data_test_list = args.data_test_path.replace(" ", "").split(",")
    for data_test in data_test_list:
        data_test_name = data_test +".csv"
        print(f"evaluating data_test_name = {data_test_name}")
        test_dataset = DistanceMapDataset(data_path=os.path.join(args.data_path, data_test_name), tokenizer=tokenizer, args=args)
        test_dataset_list.append(test_dataset)
    
    # test_dataset1 = DistanceMapDataset(data_path=os.path.join(args.data_path, "test/rna_sequences.csv"), tokenizer=tokenizer, args=args)
    # test_dataset2 = DistanceMapDataset(data_path=os.path.join(args.data_path, "DIRECT/rna_sequences.csv"), tokenizer=tokenizer, args=args)
    # test_dataset3 = DistanceMapDataset(data_path=os.path.join(args.data_path, "RFAM19/rna_sequences.csv"), tokenizer=tokenizer, args=args)


    
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
    # test_loader1 = DataLoader(test_dataset1, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
    #                          collate_fn=collate_fn)
    # test_loader2 = DataLoader(test_dataset2, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
    #                          collate_fn=collate_fn)
    # test_loader3 = DataLoader(test_dataset3, batch_size=args.per_device_eval_batch_size, shuffle=False, num_workers=args.num_workers,
    #                          collate_fn=collate_fn)

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
        wandb.init(project='DistancetMap')
        wandb.run.name = name
        wandb.run.save()
        wandb.watch(model)
        print(name)
    
    criterion = torch.nn.MSELoss()

    train_loss_batch_list = []
    train_loss_list = []
    val_loss_list = []
    test_loss_list = []
    step = 0
    last_val, best_val, best_test = -100, -100, []
    patience = args.patience
    early_stop_flag = 0
    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        for data_dict in tqdm(train_loader):
            with accelerator.accumulate(model):
                logits = model(data_dict)[:,1:-1,1:-1]
                labels = data_dict['struct']
                
                #print('label',labels.shape)
                #print('logits',logits.shape)
                label_mask = labels != -1 
                loss = criterion(logits[label_mask].reshape(-1,1), labels[label_mask].reshape(-1,1))
                #print('loss',loss)
                #print("logits dtype: ", logits.dtype)
                #print("labels dtype", labels.dtype)
                #print("Model parameters dtype: ", next(model.parameters()).dtype)
                #print("Loss dtype: ", loss.dtype)
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


        if best_val < val_metrics["r^2"]:
            best_val = val_metrics["r^2"] 
            print(f"epoch {epoch}:")
            test_metrics = []   
            for i,data_test in enumerate(data_test_list):
                test_metrics.append(test(model, test_dataloader_list[i], accelerator))
            best_test = test_metrics
        if last_val < val_metrics["r^2"]:
            early_stop_flag = 0 
        else:
            early_stop_flag +=1

        if early_stop_flag >= patience:
            print(f"Early stopping")
            break
        last_val = val_metrics["r^2"] 
        end_time = time.time()

        if accelerator.is_main_process:
            print(
                f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.2f}')
            # print(
            #     f'[VL0] Loss: {np.mean(val_loss_list)}, Top-l precision: {val_metrics["top_l_precision"]}, Top-l/2 precision: {val_metrics["top_l/2_precision"]}, Top-l/5 precision: {val_metrics["top_l/5_precision"]}, Top-l/10 precision: {val_metrics["top_l/10_precision"]}')
            # print(
            #     f'[TS0] loss: {np.mean(test_loss_list)}, Top-l precision: {test_metrics["top_l_precision"]}, Top-l/2 precision: {test_metrics["top_l/2_precision"]}, Top-l/5 precision: {test_metrics["top_l/5_precision"]}, Top-l/10 precision: {test_metrics["top_l/10_precision"]}')
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

    parser.add_argument("--pretrained_lm_dir", type=str, default='/public/home/taoshen/data/rna/mars_fm_data/mars_esm_preckpts')
    parser.add_argument('--data_path', default= '/public/home/taoshen/data/rna/mars_fm_data/downstream')
    parser.add_argument('--model_name_or_path', default='output')
    parser.add_argument('--output_dir', default='./ckpts/')
    parser.add_argument('--model_type', type=str, default='esm') # esm, esm-protein, dna
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--bprna_dir', default='/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data/')
    parser.add_argument('--run_name', type=str, default="run")
    parser.add_argument('--token_type', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--train_from_scratch', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--data_train_path', default= '/public/home/taoshen/data/rna/mars_fm_data/downstream')
    parser.add_argument('--data_val_path', default= '/public/home/taoshen/data/rna/mars_fm_data/downstream')
    parser.add_argument('--data_test_path', default= '/public/home/taoshen/data/rna/mars_fm_data/downstream')
    parser.add_argument('--attn_implementation', type=str, default="eager")
    args = parser.parse_args()

    assert args.mode in ['bprna', 'pdb']

    ## get pretrained model
    extractor, tokenizer = get_extractor(args)

    main(args)
