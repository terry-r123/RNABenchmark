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

from structure.data import SSDataset
from structure.lm import get_extractor
from structure.predictor import SSCNNPredictor

import random

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

def collate_fn(batch):
    '''

    '''
    seqs, cts, args_stack = zip(*batch)
    
    args = args_stack[0]
    # token_type = args.token_type
    # model_max_length = args.model_max_length
    # print(args)
    # print(args.model_max_length)
    # print(args.token_type)
    # for seq in seqs:
    #     print(len(seq))
    max_len = max([len(seq)+2 for seq in seqs])
    #print('max_len1',max_len)
    max_len = min(max_len, args.model_max_length)
    #print('max_len2',max_len)
    weight_mask = torch.ones((len(seqs), max_len)) #including [cls] and [sep] [bz, max_len+2]
    if args.token_type == '6mer':
        for i in range(1,5):
            weight_mask[:,i+1]=weight_mask[:,-i-2]=1/(i+1) 
        weight_mask[:, 6:-6] = 1/6
    if args.token_type == '6mer':
            seqs = [generate_kmer_str(seq, 6) for seq in seqs]
    #print(seqs[0])
    data_dict = tokenizer.batch_encode_plus(seqs, padding='longest', max_length=max_len, truncation=True, return_tensors='pt')
    position_id = torch.zeros(data_dict['attention_mask'].shape)
    #print('---------------',tokenizer)
    ## padding ct
    #print(data_dict['input_ids'].shape)
    #print(max_len)
    #print(cts[0].shape)
    #print(seqs)

    ct_masks = [np.ones(ct.shape) for ct in cts]
    cts = [np.pad(ct, (0, max_len-2-ct.shape[0]), 'constant') for ct in cts]
    ## padding ct_mask
    ct_masks = [np.pad(ct_mask, (0, max_len-2-ct_mask.shape[0]), 'constant') for ct_mask in ct_masks]
    data_dict['ct'] = torch.FloatTensor(cts)
    data_dict['ct_mask'] = torch.FloatTensor(ct_masks)
    data_dict['weight_mask'] = weight_mask
    data_dict['position_id'] = position_id

    return data_dict

def main(args):
    set_seed(args)
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers,
                              gradient_accumulation_steps=args.gradient_accumulation_steps,
                              log_with='wandb')
    model_name = args.ckpt_dir.split('/')[-1]
    #model_name = args.model_name_or_path.split('/')[-1]
    name = f'[RNA_Secondary_Structure_Prediction]{model_name}_' \
           f'{args.model_type}_' \
           f'{args.token_type}_' \
           f'lr{args.lr}_' \
           f'bs{args.batch_size}*gs{args.gradient_accumulation_steps}*gpu{accelerator.state.num_processes}_' \
           f'gs{args.gradient_accumulation_steps}' \
           f'epo{args.num_epochs}_' \
           f'warm{args.warmup_epos}epoch' \
           f'seed{args.seed}'


    if args.is_freeze:
        name += '_freeze'

    args.pdb_dir = f'{args.data_dir}/RNA_Secondary_Structure_Prediction/PDB_SS'
    #args.bprna_dir = f'{args.data_dir}/bpRNA'

    ckpt_path = os.path.join(args.ckpt_dir, name)
    os.makedirs(ckpt_path, exist_ok=True)

    
    

    model_config = extractor.config
    model = SSCNNPredictor(args, extractor, model_config, is_freeze=args.is_freeze)
    num_params = count_parameters(model)
    #print(model)
    print(f"model params are: {num_params}")
    if args.mode == 'bprna':
        ## bprna data ##
        df = pd.read_csv(f'{args.bprna_dir}/bpRNA.csv')

        df_train = df[df['data_name'] == 'TR0'].reset_index(drop=True)
        df_val = df[df['data_name'] == 'VL0'].reset_index(drop=True)
        df_test = df[df['data_name'] == 'TS0'].reset_index(drop=True)
        
        train_dataset = SSDataset(df_train, data_path=f'{args.bprna_dir}/TR0', tokenizer=tokenizer, args=args)
        val_dataset = SSDataset(df_val, data_path=f'{args.bprna_dir}/VL0', tokenizer=tokenizer, args=args)
        test_dataset = SSDataset(df_test, data_path=f'{args.bprna_dir}/TS0', tokenizer=tokenizer, args=args)
        # else:
        #     train_dataset = SSDataset(df_train, data_path=f'{args.bprna_dir}/ct/TR0', tokenizer=tokenizer, args=args)
        #     val_dataset = SSDataset(df_val, data_path=f'{args.bprna_dir}/ct/VL0', tokenizer=tokenizer, args=args)
        #     test_dataset = SSDataset(df_test, data_path=f'{args.bprna_dir}/ct/TS0', tokenizer=tokenizer, args=args)
    elif args.mode == 'pdb':
        df = pd.read_csv(f'{args.pdb_dir}/pdbRNA.csv')

        df_pdb_train = df[df['data_name']=='TR1'].reset_index(drop=True)
        df_pdb_val = df[df['data_name']=='VL1'].reset_index(drop=True)
        df_pdb_test = df[df['data_name']=='TS1'].reset_index(drop=True)

        train_dataset = SSDataset(df_pdb_train, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, args=args)
        val_dataset = SSDataset(df_pdb_val, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, args=args)
        test_dataset = SSDataset(df_pdb_test, data_path=f'{args.pdb_dir}/ct', tokenizer=tokenizer, args=args)
    
    print(f'# train: {len(train_dataset)},val:{len(val_dataset)},test:{len(test_dataset)}')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ## num_processes from accelerator
    per_steps_one_epoch = len(
        train_dataset) // args.batch_size // accelerator.num_processes // args.gradient_accumulation_steps
    num_warmup_steps = per_steps_one_epoch * args.warmup_epos
    num_training_steps = per_steps_one_epoch * args.num_epochs

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=num_warmup_steps,
                                                   num_training_steps=num_training_steps)

    model, optimizer, train_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, lr_scheduler)

    if accelerator.is_main_process:
        wandb.init(project='MARS-RNASecondaryStructure')
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
    best_val, best_test = 0, 0

    for epoch in range(args.num_epochs):
        model.train()
        start_time = time.time()
        for data_dict in tqdm(train_loader):
            with accelerator.accumulate(model):
                logits = model(data_dict)
                labels = data_dict['ct']
                #print('label',labels.shape)
                loss_list = []
                bs = logits.shape[0]
                for idx in range(bs):
                    ## exclude padding ##
                    seq_length = data_dict['attention_mask'][idx].sum().item()
                    if args.token_type == '6mer':
                        seq_length += 5
                    #print('seq_length',seq_length)
                    logit = logits[idx, :seq_length, :seq_length]
                    ## exclude padding ##
                    ## exclude start and end token ##
                    logit = logit[1:-1, 1:-1]
                    #print('logit',logit.shape)
                    #print('label',labels[idx].shape)
                    label = labels[idx, :logit.shape[0], :logit.shape[1]]
                    ## exclude start and end token ##

                    loss_list.append(criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))

                loss = torch.stack(loss_list).mean()
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

        with torch.no_grad():
            model.eval()
            for data_dict in val_loader:
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(accelerator.device)
                logits = model(data_dict)
                labels = data_dict['ct']
                loss_list = []
                bs = logits.shape[0]
                for idx in range(bs):
                    seq_length = data_dict['attention_mask'][idx].sum().item()
                    logit = logits[idx, :seq_length, :seq_length]
                    logit = logit[1:-1, 1:-1]
                    label = labels[idx, :logit.shape[0], :logit.shape[1]]
                    loss_list.append(criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))

                    probs = torch.sigmoid(logit).detach().cpu().numpy()
                    pred = (probs > threshold).astype(np.float32)
                    val_recall_list.append(recall_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                    val_precision_list.append(
                        precision_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                    val_f1_list.append(f1_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))

                loss = torch.stack(loss_list).mean()
                val_loss_list.append(loss.item())

        with torch.no_grad():
            model.eval()
            for data_dict in test_loader:
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(accelerator.device)

                logits = model(data_dict)
                labels = data_dict['ct']
                loss_list = []
                bs = logits.shape[0]
                for idx in range(bs):
                    seq_length = data_dict['attention_mask'][idx].sum().item()
                    logit = logits[idx, :seq_length, :seq_length]
                    logit = logit[1:-1, 1:-1]
                    label = labels[idx, :logit.shape[0], :logit.shape[1]]
                    loss_list.append(criterion(logit.contiguous().view(-1), label.contiguous().view(-1)))

                    probs = torch.sigmoid(logit).detach().cpu().numpy()
                    pred = (probs > threshold).astype(np.float32)
                    test_recall_list.append(recall_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                    test_precision_list.append(
                        precision_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))
                    test_f1_list.append(f1_score(label.detach().cpu().numpy().reshape(-1), pred.reshape(-1)))

                loss = torch.stack(loss_list).mean()
                test_loss_list.append(loss.item())

        if best_val < np.mean(val_precision_list) + np.mean(val_recall_list) + np.mean(val_f1_list):
            best_val = np.mean(val_precision_list) + np.mean(val_recall_list) + np.mean(val_f1_list)
            # if accelerator.is_main_process:
            #     accelerator.save_state(f'{ckpt_path}/best_val')

        if best_test < np.mean(test_precision_list) + np.mean(test_recall_list) + np.mean(test_f1_list):
            best_test = np.mean(test_precision_list) + np.mean(test_recall_list) + np.mean(test_f1_list)
            # if accelerator.is_main_process:
            #     accelerator.save_state(f'{ckpt_path}/best_test')

        end_time = time.time()

        if accelerator.is_main_process:
            print(
                f'epoch: {epoch}, lr: {optimizer.param_groups[0]["lr"]}, train_loss: {np.mean(train_loss_list):.6f}, time: {end_time - start_time:.2f}')
            print(
                f'[VL0] Loss: {np.mean(val_loss_list):.6f}, precision: {np.mean(val_precision_list):.6f}, recall: {np.mean(val_recall_list):.6f}, F1: {np.mean(val_f1_list):.6f}')
            print(
                f'[TS0] loss: {np.mean(test_loss_list):.6f}, precision: {np.mean(test_precision_list):.6f}, recall: {np.mean(test_recall_list):.6f}, F1: {np.mean(test_f1_list):.6f}')
            log_dict = {'lr': optimizer.param_groups[0]["lr"], 'train_loss': np.mean(train_loss_list)}
            log_dict.update({'VL0/loss': np.mean(val_loss_list), 'VL0/precision': np.mean(val_precision_list),
                             'VL0/recall': np.mean(val_recall_list), 'VL0/F1': np.mean(val_f1_list)})
            log_dict.update({'TS0/loss': np.mean(test_loss_list), 'TS0/precision': np.mean(test_precision_list),
                             'TS0/recall': np.mean(test_recall_list), 'TS0/F1': np.mean(test_f1_list)})
            wandb.log(log_dict)
        torch.cuda.empty_cache()
        train_loss_list, val_loss_list, test_loss_list = [], [], []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_epos', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--model_scale', type=str, default='8m')
    parser.add_argument('--is_freeze', type=bool, default=False)
    parser.add_argument('--mode', type=str, default='pdb')

    parser.add_argument("--pretrained_lm_dir", type=str, default='/public/home/taoshen/data/rna/mars_fm_data/mars_esm_preckpts')
    parser.add_argument('--data_dir', default= '/public/home/taoshen/data/rna/mars_fm_data/downstream')
    parser.add_argument('--model_name_or_path', default='./ckpts/')
    parser.add_argument('--ckpt_dir', default='./ckpts/')
    parser.add_argument('--model_type', type=str, default='esm') # esm, esm-protein, dna
    parser.add_argument('--model_max_length', type=int, default=512)
    parser.add_argument('--bprna_dir', default='/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data/')
    parser.add_argument('--non_n', type=bool, default=False)
    parser.add_argument('--token_type', type=str, default=None)
    parser.add_argument('--cache_dir', type=str, default=None)
    parser.add_argument('--train_from_scratch', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    assert args.mode in ['bprna', 'pdb']

    ## get pretrained model
    extractor, tokenizer = get_extractor(args)

    main(args)

    ## accelerate optional params
    ## --mixed_precision=fp16  ## for fp16 training
    ## --main_process_port 12306  ## for port conflict
    ## CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --multi_gpu --num_processes=8 run_train.py --num_epochs 100 --batch_size 2 --gradient_accumulation_steps 1 --lr 3e-4 --num_workers 2