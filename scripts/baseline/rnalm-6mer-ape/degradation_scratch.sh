#!/bin/bash

gpu_device="0"

nproc_per_node=1
partition='ai4multi'
USE_SLURM='2'
# 基础环境设置
MODEL_TYPE='rnalm'

task='Degradation'

# 根据 USE_SLURM 调整环境变量和执行前缀
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
    EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
fi

# 基础环境设置


for token in  'non-overlap' #'single' '6mer' 'bpe'
do
    for pos in 'ape' 'alibi' 'rope'
    do 

        MODEL_PATH=${home_root}renyuchen/RNABenchmark/model/rnalm/config/${MODEL_TYPE}-${token}-${pos}
        DATA_PATH=${data_root}multi-omics/RNA/downstream/degradation/train-val-test
        OUTPUT_PATH=./outputs/ft/rna-all/${task}/rna/baseline/${MODEL_TYPE}-${token}-${pos}-scratch
        batch_size=32
        data=''
        data_file_train=train_1.json; data_file_val="val_1.json"; data_file_test="test_1.json"
        for seed in 42 666 3407
        do

            master_port=$(shuf -i 10000-45000 -n 1)
            echo "Using port $master_port for communication."
            echo ${MODEL_PATH}
            EXEC_PREFIX="srun --nodes=1 -p ${quotatype} -A ${partition} --job-name=${MODEL_TYPE}_${task}  --gres=gpu:$nproc_per_node --cpus-per-task=32 torchrun --nproc_per_node=$nproc_per_node --master_port=$master_port"
            ${EXEC_PREFIX} \
                downstream/train_degradation.py \
                    --model_name_or_path ${MODEL_PATH} \
                    --data_path  ${DATA_PATH}/${data} \
                    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
                    --run_name ${MODEL_TYPE}_${data}_seed${seed} \
                    --model_max_length 1026 \
                    --per_device_train_batch_size $batch_size \
                    --per_device_eval_batch_size 32 \
                    --gradient_accumulation_steps 1 \
                    --learning_rate 3e-5 \
                    --num_train_epochs 1 \
                    --fp16 \
                    --save_steps 400 \
                    --output_dir ${OUTPUT_PATH} \
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

                    for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do export $v=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128; done 
                    
                    kaggle competitions submit -c stanford-covid-vaccine -f ${OUTPUT_PATH}/results/${MODEL_TYPE}_${data}_seed${seed}/submission_rnalm-${token}-${pos}-scratch.csv -m "Message" \


        done
    done
done
