#!/bin/bash
# Example Slurm job for the 23-setting x 9-method x 10-seed evaluation (2,070 episodes).
# Replace every <PLACEHOLDER> below with your cluster values before use.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ppr01_DRIVER
#SBATCH --time=3-00:00:00
#SBATCH --partition=<PARTITION>
#SBATCH --account=<ACCOUNT>
#SBATCH --chdir=<PROJECT_ROOT>
#SBATCH --output=logs/driver.%j.out.txt
#SBATCH --error=logs/driver.%j.err.txt

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
srun --ntasks=1 --cpus-per-task=1 python -u driver.py \
  --dataset-dir DATASET \
  --model-dir models \
  --output-dir results \
  --workbook PPR01_V209_EVALUATION_job9851603.xlsx
conda deactivate
