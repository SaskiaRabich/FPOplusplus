#!/bin/bash
# call from root directory of FPOplusplus

# data
scene=sport_1
dataset=nhr
exp_name="nhr_sport1"
timesteps=60
timeskip=1
timeoffset=0

# input and output files
name_fpo=fpo
name_fpo_finetuned=fpo_opt

# fourier plenoctree
num_epochs=10
val_interval=1

python fpo/finetune_dynamic_tree.py --train_dir ./logs/$exp_name/octrees --config ./fpo/config/$dataset --data_dir ./data/$dataset/$scene/ --tree_name $name_fpo --save_name $name_fpo_finetuned --time_steps $timesteps --time_skip $timeskip --time_offset $timeoffset --num_epochs $num_epochs --val_interval $val_interval --log_file_train mse_train_$name_fpo_finetuned.txt --log_file_test mse_test_$name_fpo_finetuned.txt
