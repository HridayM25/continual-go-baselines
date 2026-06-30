#!/bin/bash
#SBATCH --account=aip-mtaylor3
#SBATCH --job-name=c_go_w2
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/home/hriday/scratch/cont_go/logs/ord_%A_%a.out
#SBATCH --error=/home/hriday/scratch/cont_go/logs/ord_%A_%a.err
# ---------------------------------------------------------------------------
PROJECT_ROOT=/project/aip-mtaylor3/hriday/continual-go
mkdir -p /home/hriday/scratch/cont_go/logs
export PATH="$HOME/.local/bin:$PATH"      # ensure uv is found
export PYTHONPATH=$PROJECT_ROOT           # so alpha_zero / continual_go resolve
export UV_NO_SYNC=1                        # don't hit the network on a compute node
export UV_LINK_MODE=copy                   # silence the hardlink warning

# export PYTHONUNBUFFERED=1
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"

uv add "torch" --index https://download.pytorch.org/whl/cpu


cd $PROJECT_ROOT
uv run python -u algorithms/train.py 
EXIT_CODE=$?
echo "=== done | exit $EXIT_CODE | $(date) ==="
exit $EXIT_CODE