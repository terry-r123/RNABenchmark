#!/bin/bash
export kmer=-1
export MODEL_PATH=/mnt/data/oss_beijing/multi-omics/RNA/model/opensource/rna-fm
export DATA_PATH=/mnt/data/oss_beijing/multi-omics/RNA/downstream/degradation/train-val-test
export OUTPUT_PATH=/mnt/data/oss_beijing/renyuchen/temp/ft/rna-all/egradation/rna/opensource/rna-fm

echo "The provided data_path is $DATA_PATH"
export gpu_device="1"
export master_port=33333
export nproc_per_node=1
export kmer=-1
export token='single'
export MODEL_TYPE='rna-fm'
export STAGE=None
export batch_size=32
export data=''
export lr=3e-5
for seed in 42 666 3407
do
    CUDA_VISIBLE_DEVICES=$gpu_device torchrun \
            --nproc_per_node $nproc_per_node \
            --master_port $master_port \
        downstream/train_degradation.py \
        --model_name_or_path ${MODEL_PATH} \
        --data_path ${DATA_PATH}/${data} \
        --kmer ${kmer} \
        --run_name ${MODEL_TYPE}${data}_seed${seed} \
        --model_max_length 512 \
        --per_device_train_batch_size ${batch_size} \
        --per_device_eval_batch_size 32 \
        --gradient_accumulation_steps 1 \
        --learning_rate ${lr} \
        --num_train_epochs 100 \
        --fp16 \
        --save_steps 200 \
        --output_dir ${OUTPUT_PATH}/${data} \
        --log_dir ${OUTPUT_PATH}/${MODEL_TYPE}${data}_seed${seed} \
        --seed ${seed} \
        --save_model True \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --warmup_steps 50 \
        --logging_steps 200 \
        --overwrite_output_dir True \
        --log_level info \
        --find_unused_parameters False \
        --eval_accumulation_steps 1 \
        --lr_scheduler_type cosine_with_restarts \
        --model_type ${MODEL_TYPE} \
        --token_type ${token} \

    kaggle competitions submit -c stanford-covid-vaccine -f ${OUTPUT_PATH}/results/${MODEL_TYPE}${data}_seed${seed}/submission_.csv -m "Message"

done