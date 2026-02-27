#!/bin/bash
#SBATCH -A bfsr-delta-cpu 
#SBATCH --job-name="extract"
#SBATCH --output="extract.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 200000
#SBATCH -t 0-2
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
input_path="${base_path}/raw_video/${eid}-${camera}"
output_path="${base_path}/beast/extracted_frames/${eid}"

echo "Output will be saved to: $output_path"

# Change to repo root
cd ..

# Activate environment
conda activate brainwide

beast extract --frames-per-video 2000 \
  --input "$input_path" \
  --output "$output_path"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
