#!/bin/bash
# call from root directory of FPOplusplus

# data
exp_name=nhr_sport1
scene=sport_1
dataset=nhr

# fourier plenoctree
tree_name=fpo_opt
timesteps=60
timeskip=1
timeoffset=0

#write_images="--write_images ./logs/$exp_name/octrees/imgs/" # uncomment to write test images to the given directory

python ./fpo/evaluate_dynamic_tree.py --input ./logs/$exp_name/octrees/$tree_name.npz --config ./fpo/config/$dataset --data_dir ./data/$dataset/$scene/ --time_steps $timesteps --time_skip $timeskip --time_offset $timeoffset $write_images

