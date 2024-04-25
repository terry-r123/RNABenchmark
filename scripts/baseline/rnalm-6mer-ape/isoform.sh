#!/bin/bash

export gpu_device="1"
export master_port=41707
export nproc_per_node=1

# 根据 USE_SLURM 调整环境变量和执行前缀
if [ "$USE_SLURM" == "1" ]; then
    export batch_size=32
    export nproc_per_node=1
    export master_port=41707
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    export batch_size=16
    export nproc_per_node=2
    export master_port=41708
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_alt_${data} --gres=gpu:2 --cpus-per-task=10 --mem=100G --qos=highpriority"
else
    export data_root=/mnt/data/oss_beijing/
    export CUDA_VISIBLE_DEVICES=$gpu_device
    EXEC_PREFIX="torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
fi

# 基础环境设置
export MODEL_TYPE='rnalm'
export token='6mer'
for pos in 'ape' 'alibi' 'rope'
do 

    export MODEL_PATH=/mnt/data/ai4bio/renyuchen/RNABenchmark/model/rnalm/config/${MODEL_TYPE}-${token}-${pos}
    export DATA_PATH=${data_root}multi-omics/RNA/downstream/Isoform
    export OUTPUT_PATH=./outputs/ft/rna-all/Isoform/rna/baseline/${MODEL_TYPE}-${token}-${pos}-scratch
    export batch_size=32
    export CRYPTOGRAPHY_OPENSSL_NO_LEGACY=1
    export data=''

    for seed in 42 666 3407
    do

        echo ${MODEL_PATH}

        CUDA_VISIBLE_DEVICES=$gpu_device torchrun \
            --nproc_per_node $nproc_per_node \
            --master_port $master_port \
            downstream/train_isoform.py \
                --model_name_or_path ${MODEL_PATH} \
                --data_path  ${DATA_PATH}/${data} \
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
                --train_from_scratch True
    done
    
done
