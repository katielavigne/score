#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --account=def-mlepage
#SBATCH --job-name=apply_score
#SBATCH --output=%x-%j.out

# BEFORE RUNNING THIS SCRIPT (do this once):
# create ENV w/ statsmodels & bctpy packages

# ENV="$SCRATCH/score/scoreENV"
# virtualenv --no-download $ENV
# source $ENV/bin/activate
# pip install --upgrade pip
# pip install statsmodels --no-index
# pip install bctpy --no-index

# MODIFY LINES 104-111 of apply_score.py to fit your data

# THEN RUN THIS SCRIPT: sbatch run_apply_score.sh

module load python scipy-stack/2022a

source $SCRATCH/score/scoreENV/bin/activate

python apply_score.py