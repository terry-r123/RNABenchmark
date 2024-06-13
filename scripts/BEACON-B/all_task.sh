#!/bin/bash

# This is your argument

gpu_device="0"

nproc_per_node=1
master_port=$(shuf -i 10000-45000 -n 1)
echo "Using port $master_port for communication."



data_root=./data
model_root=./checkpoint



MODEL_TYPE='rnalm'

token='single'
pos='alibi'



model_max_length=1026
seed=666
data=''
MODEL_PATH=${model_root}/baseline/BEACON-B/



        
task='Secondary_structure_prediction'
batch_size=1
lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_secondary_structure.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path  ${DATA_PATH}/${data} \
    --run_name ${MODEL_TYPE}_${data} \
    --output_dir ${OUTPUT_PATH}/${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr ${lr} \
    --num_epochs 100 \
    --patience 60 \
    --num_workers 1 \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \
    --seed ${seed} \





task='ContactMap'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test,RFAM19,DIRECT
batch_size=1

lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_contact_map.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path  ${DATA_PATH}/${data} \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --output_dir ${OUTPUT_PATH}/${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr ${lr} \
    --num_epochs 100 \
    --patience 60 \
    --num_workers 1 \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \
    --seed ${seed} \
    


task='DistanceMap'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test,RFAM19,DIRECT
batch_size=1
lr=5e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_distance_map.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path  ${DATA_PATH}/${data} \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --output_dir ${OUTPUT_PATH}/${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr ${lr} \
    --num_epochs 100 \
    --patience 60 \
    --num_workers 1 \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \
    --seed ${seed} \


task='StructuralScoreImputation'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_structural_score_imputation.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
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


task='SpliceAI'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_spliceai.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
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

task='Isoform'
data_file_train=train_new.csv; data_file_val=val.csv; data_file_test=test
batch_size=32
lr=5e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_isoform.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
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

task='NoncodingRNAFamily'
data_file_train=train_new.csv; data_file_val=val.csv; data_file_test=test
batch_size=16
lr=5e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_ncrna.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \


task='Modification'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_modification.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data}_seed${seed}_lr${lr} \
    --model_max_length 1026 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \


task='MeanRibosomeLoading'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=1e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_mean_ribosome_loading.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length 1026 \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \


task='Degradation'
data_file_train=train_1.json; data_file_val=val_1.json; data_file_test=test_1.json
batch_size=32
lr=5e-5
DATA_PATH=${data_root}/downstream/${task}/train-val-test
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_degradation.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 100 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \


task='ProgrammableRNASwitches'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=1e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_programmable_rna_switches.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \

task='CRISPROnTarget'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=1e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_crispr_on_target.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
    --fp16 \
    --save_steps 400 \
    --output_dir ${OUTPUT_PATH}/${seed} \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --warmup_steps 50 \
    --logging_steps 200 \
    --overwrite_output_dir True \
    --log_level info \
    --seed ${seed} \
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \

task='CRISPROffTarget'
data_file_train=train.csv; data_file_val=val.csv; data_file_test=test.csv
batch_size=32
lr=3e-5
DATA_PATH=${data_root}/downstream/${task}
OUTPUT_PATH=./outputs/ft/rna-all/${task}/BEACON-B/${MODEL_TYPE}  
EXEC_PREFIX="env CUDA_VISIBLE_DEVICES=$gpu_device torchrun --num_processes=$nproc_per_node --main_process_port=$master_port --mixed_precision=fp16"
echo ${MODEL_PATH}
${EXEC_PREFIX} \
downstream/train_crispr_off_target.py \
    --model_name_or_path $MODEL_PATH \
    --data_path  $DATA_PATH/$data \
    --data_train_path ${data_file_train} --data_val_path ${data_file_val} --data_test_path ${data_file_test}   \
    --run_name ${MODEL_TYPE}_${data} \
    --model_max_length ${model_max_length} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate ${lr} \
    --num_train_epochs 30 \
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
    --token_type ${token} \
    --model_type ${MODEL_TYPE} \