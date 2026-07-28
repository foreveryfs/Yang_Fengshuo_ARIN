#!/bin/bash
#SBATCH --account=mscaisuperpod
#SBATCH --job-name=run_train
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=4:00:00
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err

module purge

source /home/fyangbe/miniconda3/etc/profile.d/conda.sh  
conda activate MIA

cd ~/MIA/CoW

# bash ./scripts/train_abd_mri.sh
bash ./scripts/train_abd_ct.sh
# bash ./scripts/train_cmr_mri.sh


