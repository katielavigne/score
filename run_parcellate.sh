#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=80G
#SBATCH --account=def-mlepage
#SBATCH --job-name=parcellate
#SBATCH --output=%x-%j.out

# sbatch run_parcellate.sh

module load python scipy-stack/2022a

python parcellate.py
