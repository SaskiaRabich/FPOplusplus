#!/bin/bash
# call from root directory of FPOplusplus

exp_name=nhr_sport1
tree_file=fpo_opt

./volrend/build/volrend ./logs/$exp_name/octrees/$tree_file.npz
