#!/bin/bash

gpu_device="0"
master_port=41611
nproc_per_node=1
partition='ai4bio'
USE_SLURM='0'
# 基础环境设置
MODEL_TYPE='rnalm'

task='Secondary_structure_prediction'

# 根据 USE_SLURM 调整环境变量和执行前缀
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    uotatype='vip_gpu_ailab'
    module load anaconda/2021.11
    module load cuda/11.7.0
    module load cudnn/8.6.0.163_cuda11.x
    module load compilers/gcc/9.3.0
    module load llvm/triton-clang_llvm-11.0.1
    source activate dnalm_v2
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/peng_tmp_test/miniconda3/lib
    export CPATH=/usr/include:$CPATH
    export PYTHONUNBUFFERED=1
    export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so.6
    data_root=/home/bingxing2/ailab/group/ai4bio/public/
    EXEC_PREFIX="srun --nodes=1 -p $quotatype -A $partition --job-name=${MODEL_TYPE}_${task} --gres=gpu:$nproc_per_node --cpus-per-task=32 torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
else
    data_root=/mnt/data/oss_beijing/   
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
fi
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

