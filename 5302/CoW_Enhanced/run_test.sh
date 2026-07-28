#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --job-name=run_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=2:00:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

module purge

source /home/fyangbe/miniconda3/etc/profile.d/conda.sh  
conda activate MIA

cd ~/MIA/CoW

# bash ./scripts/train_abd_mri.sh
bash ./scripts/test_abd_ct.sh
# bash ./scripts/train_cmr_mri.sh


