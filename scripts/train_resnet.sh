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
#SBATCH --time 1-00:00
#SBATCH --export=ALL

eid=${1:-none}
camera=${2:-none}

if [ "$eid" = "none" ]; then
  echo "Error: Eid not provided." >&2
  exit 1
fi

if [ "$camera" = "none" ]; then
  echo "Error: Camera not provided." >&2
  exit 1
fi

# Load environment
. ~/.bashrc

module load ffmpeg

# Configuration
dataset_name="brainwidemap"
account_name="bezq"
username="$(whoami)"

base_path="/work/nvme/${account_name}/${username}/${dataset_name}"
data_path="${base_path}/beast/extracted_frames/${eid}-${camera}"
checkpoint_path="${base_path}/resnet/checkpoints/${eid}-${camera}"

echo "Output will be saved to: $base_path"

# Change to repo root
cd ..

# Activate environment
conda activate brainwide

beast train --config configs/resnet_ae.yaml \
  --data "$data_path" \
  --output "$checkpoint_path"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
