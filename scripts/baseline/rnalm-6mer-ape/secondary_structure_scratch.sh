#!/bin/bash

# This is your argument
export kmer=6
export MODEL_PATH=/mnt/data/ai4bio/renyuchen/DNABERT/examples/output/rna/base/noncoding_rna_human_6mer_1024/checkpoint-80000
export DATA_PATH=/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data 
export bprna_PATH=/mnt/data/ai4bio/rna/downstream/Secondary_structure_prediction/esm_data/bpRNA
export OUTPUT_PATH=./outputs/ft/rna-all/secondary_structure/rna/baseline/rnalm-6mer-ape/scratch
export STAGE=None
export MODEL_TYPE='rnabert'
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
--batch_size 1 \
--gradient_accumulation_steps 2 \
--lr 3e-4 \
--num_workers 1 \
--token_type '6mer' \
--model_max_length 1026 \
--non_n True \
--train_from_scratch True \

