#!/usr/bin/sh

data_src=${1-'../data'}
data_dest=${2-'../pre_data'}
num_samples=${3-5000}

mkdir -p $data_dest

python ../Datasets/Kitti_loader.py --num_samples $num_samples --datapath $data_src --dest $data_dest

# copy non existent files over (ground truth etc)
cp -r -n $data_src/* $data_dest
