#!/bin/bash
# call from root directory of FPOplusplus

# data
exp_name="nhr_sport1"
timesteps=60
timeskip=1
timeoffset=0

# input and output files
name_plenoctree=tree
name_fpo=fpo

# fourier plenoctree
data_format="LFC"
comp_encoding=True
augment_time=True
sh_dim=9
fc_dim_sigma=31
fc_dim_rgb=5

python fpo/construct_dynamic_tree.py --train_dir ./logs/$exp_name --output ./logs/$exp_name/octrees --tree_name $name_plenoctree --save_name $name_fpo --time_steps $timesteps --time_skip $timeskip --time_offset $timeoffset --sh_dim $sh_dim --fc_dim_sigma $fc_dim_sigma --fc_dim_rgb $fc_dim_rgb --data_format $data_format --augment_time=$augment_time --comp_enc=$comp_encoding
