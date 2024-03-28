#!/bin/bash

# This is your argument
export kmer=-1
export MODEL_PATH=/mnt/data/oss_beijing/multi-omics/RNA/model/opensource/rna-fm
export DATA_PATH=/mnt/data/oss_beijing/multi-omics/RNA/downstream/ProgrammableRNASwitches
export OUTPUT_PATH=./outputs/ft/rna-all/ProgrammableRNASwitches/rna/opensource/rna-fm
export STAGE=None
export MODEL_TYPE=rna-fm
export gpu_device="1"
export master_port=41452
export nproc_per_node=1
export batch_size=32
export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
export token='single'
export data=''
for seed in 42 666 3407
do

    

    CUDA_VISIBLE_DEVICES=$gpu_device torchrun \
        --nproc_per_node $nproc_per_node \
        --master_port $master_port \
        downstream/train_programmable_rna_switches.py \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH/$data \
            --kmer ${kmer} \
            --run_name ${MODEL_TYPE}_${data}_seed${seed} \
            --model_max_length 1026 \
            --per_device_train_batch_size $batch_size \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate 3e-5 \
            --num_train_epochs 30 \
            --fp16 \
            --save_steps 400 \
            --output_dir ${OUTPUT_PATH}/${data} \
            --evaluation_strategy steps \
            --eval_steps 200 \
            --warmup_steps 50 \
            --logging_steps 200 \
            --overwrite_output_dir True \
            --log_level info \
            --seed ${seed} \
            --find_unused_parameters False \
            --stage $STAGE \
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
    
done
