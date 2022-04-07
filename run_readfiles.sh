#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=249G
#SBATCH --account=def-mlepage
#SBATCH --job-name=readfiles
#SBATCH --output=%x-%j.out

# sbatch run_readfiles.sh

module load python scipy-stack/2022a

python readfiles.py
