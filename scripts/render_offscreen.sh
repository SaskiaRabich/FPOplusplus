#!/bin/bash
# call from root directory of FPOplusplus

# data
exp_name=nhr_sport1
output_dir=./logs/$exp_name/volrend_headless
pose_dir="" # path to directory with camera poses
intr_file="" # path to file with camera intrinsics

# fourier plenoctree
tree_file=fpo_opt
measure_FPS=false # if true, will only render the images without writing them to output_dir
timestep=0 # if rendering multiple time steps, this is the first to render
num=60 # 1 for only rendering "timestep", >1 for more
skip=1 # set to n for rendering only every n-th time step


if [ -z "${pose_dir}" ];
then
    echo "Variable 'pose_dir' is not set but is required for offscreen rendering. Enter the path to the directory containing the camera poses, exiting."
    exit 0
fi
if [ -z "${intr_file}" ];
then
    echo "Variable 'intr_file' is not set but is required for offscreen rendering. Enter the path to the intrinsics file, exiting."
    exit 0
fi

for (( c=$timestep; c<$num; c+=$skip ))
do
    echo "Time step: $c"
    output_dir_tmp=$output_dir/$tree_file/$c
    if [ "$measure_FPS" == true ] ;
    then
        ./volrend/build/volrend_headless ./logs/$exp_name/octrees/$tree_file.npz -i $intr_file $pose_dir/* -t $c 
    else
        ./volrend/build/volrend_headless ./logs/$exp_name/octrees/$tree_file.npz -i $intr_file $pose_dir/* -o $output_dir_tmp -t $c
    fi
done
