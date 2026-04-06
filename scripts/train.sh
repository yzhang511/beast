#!/bin/bash
#SBATCH --account=bfsr-delta-gpu
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --job-name="train"
#SBATCH --output="train.%j.out"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100000
#SBATCH --time 0-08:00
#SBATCH --export=ALL

data_path=${1}
checkpoint_path=${2}
config_path=${3}

# Load environment
. ~/.bashrc

module load ffmpeg

echo "Output will be saved to: $checkpoint_path"

# Change to repo root
cd ..

# Activate environment
conda activate beast

beast train --config "$config_path" \
  --data "$data_path" \
  --output "$checkpoint_path"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
