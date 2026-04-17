#!/bin/bash
#SBATCH -A bfsr-delta-cpu 
#SBATCH --job-name="beast_extract"
#SBATCH --output="scripts/beast_extract.%j.out"
#SBATCH --error="scripts/beast_extract.%j.err"
#SBATCH --partition=cpu
#SBATCH -c 1
#SBATCH --mem 10G
#SBATCH -t 0-2:00:00
#SBATCH --export=ALL

source ~/.bashrc
conda activate beast

STAGE=eval


DATASET_BASE=/work/nvme/bfsr/xdai3/IBL_data/synchronized/IBL-2view
OUTPUT_BASE=/work/nvme/bfsr/xdai3/IBL_data/synchronized/extracted_frames
DATASET_LIST=$DATASET_BASE/opencv_cameras_pairs_3_sessions.txt
LITPOSE_CORRESPONDENCES_ROOT=$DATASET_BASE/litpose_correspondences

beast extract \
  --input $DATASET_BASE/leftCamera.video \
  --output $OUTPUT_BASE/eval/leftCamera.video \
  --method timestamp \
  --timestamp_dir $DATASET_BASE/timestamps \
  --neural_data_dir /work/nvme/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data \
  --eid ecb5520d-1358-434c-95ec-93687ecd1396

beast extract \
  --input $DATASET_BASE/rightCamera.video \
  --output $OUTPUT_BASE/eval/rightCamera.video \
  --method timestamp \
  --timestamp_dir $DATASET_BASE/timestamps \
  --neural_data_dir /work/nvme/bfsr/xdai3/IBL_data/synchronized/extracted_frames/neural_data \
  --eid ecb5520d-1358-434c-95ec-93687ecd1396

conda deactivate