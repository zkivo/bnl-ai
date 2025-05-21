#!/bin/bash
#SBATCH -t 04:00:00

#SBATCH -J BNL-AI

#SBATCH -o log/%j.out
#SBATCH -e log/%j.err

#SBATCH -p gpua100

python3 marco/train_pose.py