#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --mem=249G
#SBATCH --account=def-mlepage
#SBATCH --job-name=score
#SBATCH --output=%x-%j.out

# BEFORE RUNNING THIS SCRIPT (do this once):
# create ENV w/ statsmodels & bctpy packages

# ENV="$SCRATCH/score/scoreENV"
# virtualenv --no-download $ENV
# source $ENV/bin/activate
# pip install --upgrade pip
# pip install statsmodels --no-index
# pip install bctpy --no-index

# MODIFY LINES 111-118 of score.py to fit your data

# THEN RUN THIS SCRIPT: sbatch run_score.sh

module load python scipy-stack/2022a

source $SCRATCH/score/scoreENV/bin/activate

python score.py
