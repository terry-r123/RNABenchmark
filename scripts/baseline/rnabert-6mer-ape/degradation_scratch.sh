#!/bin/bash

data_path=/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test
lr=5e-5
#model_name=/mnt/data/oss_beijing/liangchaoqi/sjiqun/dna_pretrained_output/pre2048-multi-all-csv-dnabert2-6-mer-token-all-mask-correct-rate-0.025/checkpoint-1000000
#model_name=/mnt/data/oss_beijing/baiweiqiang/dnamodel/pre2048-multi_1KG-13M_bpe-shuffle-dnabert/checkpoint-400000
model_name=/mnt/data/ai4bio/renyuchen/DNABERT/examples/output/rna/base/noncoding_rna_human_6mer_1024/checkpoint-80000
output_name=/mnt/data/oss_beijing/renyuchen/temp/ft/rna-all/degra/rna/base/rna-6mer-ape-human-scratch
train_file=train_rna_degradation.py
use_alibi=True

echo "The provided data_path is $data_path"
export gpu_device="1"
export master_port=33333
export nproc_per_node=1
export kmer=6
export token='6mer'
export MODEL_TYPE='rnabert'
export STAGE=None
export batch_size=32
export data=''

for seed in 42 666 3407
do
    CUDA_VISIBLE_DEVICES=$gpu_device torchrun \
            --nproc_per_node $nproc_per_node \
            --master_port $master_port \
        downstream/train_degradation.py \
        --model_name_or_path ${model_name} \
        --data_path  $data_path \
        --kmer ${kmer} \
        --use_alibi ${use_alibi} \
        --run_name ${MODEL_TYPE}${data}_seed${seed} \
        --model_max_length 512 \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 100 \
        --fp16 \
        --save_steps 200 \
        --output_dir ${output_name} \
        --log_dir ${output_name}/${MODEL_TYPE}_${data}_seed${seed} \
        --seed ${seed} \
        --save_model True \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 500 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_accumulation_steps 1 \
        --lr_scheduler_type cosine_with_restarts \
        --model_type ${MODEL_TYPE} \
        --token_type ${token} \
        --train_from_scratch True

done