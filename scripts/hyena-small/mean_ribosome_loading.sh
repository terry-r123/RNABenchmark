#!/bin/bash

gpu_device="1"
master_port=40118
nproc_per_node=1

# 根据 USE_SLURM 调整环境变量和执行前缀
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then

    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_alt_${data} --gres=gpu:2 --cpus-per-task=10 --mem=100G --qos=highpriority"
else
    data_root=/mnt/data/oss_beijing/
    CUDA_VISIBLE_DEVICES=$gpu_device
    EXEC_PREFIX="torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
fi

# 基础环境设置
MODEL_TYPE='hyenadna'
token='single'
task='MeanRibosomeLoading'

MODEL_PATH=${data_root}multi-omics/DNA/model/opensource/hyena/hf/hyenadna-small-32k-seqlen-hf/
DATA_PATH=${data_root}multi-omics/RNA/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/dna/opensource/${MODEL_TYPE}/${MODEL_TYPE}-small
batch_size=32
CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
data=''
data_file_train=train.csv; data_file_val="val.csv"; data_file_test="test.csv"
for seed in 42 666 3407
do

    echo ${MODEL_PATH}

    EXEC_PREFIX \
        downstream/train_mean_ribosome_loading.py \
            --model_name_or_path ${MODEL_PATH} \
            --data_path  ${DATA_PATH}/${data} \
            --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
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
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
            
done

