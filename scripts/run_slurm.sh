#!/bin/bash -l
#SBATCH --job-name="navierstokes3D"
#SBATCH --output=navierstokes3D.%j.o
#SBATCH --error=navierstokes3D.%j.e
#SBATCH --time=03:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account class04

# Example job script for a multi-GPU run on a CUDA-aware MPI cluster (adapt modules/account).
module load daint-gpu
module load Julia

export MPICH_RDMA_ENABLED_CUDA=1
export IGG_CUDAAWARE_MPI=1
export NS3D_USE_GPU=true

srun -n8 bash -c 'LD_PRELOAD="/usr/lib64/libcuda.so:/usr/local/cuda/lib64/libcudart.so" julia --project -O3 --check-bounds=no scripts/run.jl 255 1000 out'
