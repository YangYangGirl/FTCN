#!/bin/bash

#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=90GB
#PBS -q gpuvolta
#PBS -P kf09
#PBS -l walltime=20:00:00
#PBS -l storage=gdata/kf09+scratch/kf09
#PBS -l wd


export CONDA_ENV='/scratch/kf09/yy8664/miniconda3/bin/activate'
source $CONDA_ENV ftcn

python3 train_hdf5_v4_ftcn.py src/configs/ftcn/base.json -n train_hdf5_v4 > logs/train_hdf5_v4.log 2>&1
