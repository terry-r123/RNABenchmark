#!/bin/bash

# This is your argument
export kmer=6
export MODEL_PATH=/mnt/data/oss_beijing/multi-omics/RNA/model/opensource/rna-fm
export DATA_PATH=/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data 
export bprna_PATH=/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data/bpRNA
export OUTPUT_PATH=./outputs/ft/rna-all/secondary_structure/rna/opensource/rna-fm
export STAGE=None
export MODEL_TYPE='rna-fm'
export gpu_device=0
export master_port=40123


CUDA_VISIBLE_DEVICES=1  accelerate launch --num_processes=1 --main_process_port=${master_port} \
downstream/train_secondary_structure.py \
--model_type ${MODEL_TYPE} \
--mode bprna \
--data_dir ${DATA_PATH} \
--bprna_dir ${bprna_PATH} \
--model_name_or_path ${MODEL_PATH} \
--ckpt_dir ${OUTPUT_PATH} \
--num_epochs 100 \
--batch_size 2 \
--gradient_accumulation_steps 4 \
--lr 3e-4 \
--num_workers 1 \
--token_type 'single' \
--model_max_length 1026 \



