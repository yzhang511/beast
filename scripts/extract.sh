#!/bin/bash
#SBATCH -A bezq-delta-cpu 
#SBATCH --job-name="extract"
#SBATCH --output="extract.%j.out"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 200000
#SBATCH -t 0-03:00:00
#SBATCH --export=ALL

# Load environment
. ~/.bashrc

module load ffmpeg

input_path=${1}
output_path=${2}
method=${3}
frames_per_video=${4}

echo "Output will be saved to: $output_path"

# Change to repo root
cd ..

# Activate environment
conda activate beast

beast extract --input "$input_path" \
  --output "$output_path" \
  --method "$method" \
  --frames-per-video "$frames_per_video"

# Deactivate environment
conda deactivate

# Return to scripts directory
cd scripts
