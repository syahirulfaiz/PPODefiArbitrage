#!/bin/bash
# Example Slurm job for training PPO/SAC/TD3.
# Replace every <PLACEHOLDER> below with your cluster values before use.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --job-name=ppr01_TRAIN
#SBATCH --time=3-00:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --chdir=<PROJECT_ROOT>
#SBATCH --output=logs/train.%j.out.txt
#SBATCH --error=logs/train.%j.err.txt

# Create logs/ before sbatch; Slurm opens the log files before running this script.
set -eo pipefail
module purge
module load miniconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate <CONDA_ENV>
set -u
cd "<PROJECT_ROOT>"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1
python --version
srun --ntasks=1 --cpus-per-task=3 python -u train.py \
  --dataset-dir DATASET \
  --output-dir models \
  --trials 12 \
  --hpo-steps 100000 \
  --steps 1000000 \
  --jobs 3
conda deactivate
