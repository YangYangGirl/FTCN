#!/bin/bash

#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32GB
#PBS -q gpuvolta
#PBS -P kf09
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/kf09+scratch/kf09
#PBS -l wd


export CONDA_ENV='/scratch/kf09/yy8664/miniconda3/bin/activate'
source $CONDA_ENV ftcn

python test_on_raw_video_CDF.py examples/shining.mp4 output -d CDF > logs/test.log 2>&1

# python test_on_raw_video_CDF.py examples/shining.mp4 output -d CDF > logs/test.log 2>&1