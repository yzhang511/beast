#!/bin/bash

ONE_CACHE_PATH=${1}
VIDEO_TIMESTAMPS=${2}
OUTPUT_PATH=${3}
NUM_TRIALS=${4}

# List EIDs here and run this script with no arguments
EIDS=(
  "4b00df29-3769-43be-bb40-128b1cba6d35"
  "3e6a97d3-3991-49e2-b346-6948cb4580fb"
  "5dcee0eb-b34d-4652-acc3-d10afc6eae68"
  "72cb5550-43b4-4ef0-add5-e4adfdfb5e02"
  "781b35fd-e1f0-4d14-b2bb-95b7263082bb"
)

for eid in "${EIDS[@]}"; do
  echo "Submitting job for eid=${eid}"
  sbatch extract_neural_data.sh "$eid" "$ONE_CACHE_PATH" "$VIDEO_TIMESTAMPS" "$OUTPUT_PATH" "$NUM_TRIALS"
done
