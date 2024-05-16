#!/bin/bash

# This is your argument

gpu_device="2"

nproc_per_node=1
partition='ai4multi'
USE_SLURM='2'

MODEL_TYPE='rna-fm'

task='MeanRibosomeLoading'
token='single'
pos='ape'
export CUDA_LAUNCH_BLOCKING=1
if [ "$USE_SLURM" == "1" ]; then
    EXEC_PREFIX="srun --job-name=${MODEL_TYPE}_${data} --gres=gpu:$nproc_per_node --cpus-per-task=$(($nproc_per_node * 5)) --mem=50G"
elif [ "$USE_SLURM" == "2" ]; then
    quotatype='vip_gpu_ailab'
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
    home_root=/home/bingxing2/ailab/group/ai4bio/
    
else
    data_root=/mnt/data/oss_beijing/   
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port"
fi

DATA_PATH=${data_root}multi-omics/RNA/downstream/${task}
batch_size=32
data=''
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
MODEL_PATH=/home/bingxing2/ailab/group/ai4bio/public/huggingface/hub/models--multimolecule--rnafm/snapshots/497cde86e77d6e7059b6fe2115e1c78896d618f9 #${data_root}multi-omics/RNA/model/opensource/${MODEL_TYPE}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/opensource/${MODEL_TYPE}  


        
for seed in 42 666 3407
do

    for lr in  3e-5 #5e-5 1e-5 5e-4 1e-4 5e-6 1e-6
    do 

        
         
        master_port=$(shuf -i 10000-45000 -n 1)
        echo "Using port $master_port for communication."
        EXEC_PREFIX="sbatch --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=32 torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"           
        echo ${MODEL_PATH}

        ${EXEC_PREFIX} \
        downstream/train_mean_ribosome_loading.py \
            --model_name_or_path $MODEL_PATH \
            --data_path  $DATA_PATH/$data \
            --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
            --run_name ${MODEL_TYPE}_${data}_seed${seed}_lr${lr} \
            --model_max_length 1026 \
            --per_device_train_batch_size ${batch_size} \
            --per_device_eval_batch_size 32 \
            --gradient_accumulation_steps 1 \
            --learning_rate ${lr} \
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
            --token_type ${token} \
            --model_type ${MODEL_TYPE} \
 
    done
done


