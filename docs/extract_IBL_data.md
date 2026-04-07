## Extract IBL data

#### Extract video frames for model fine-tuning

(< 0.25 hours on CPU) To extract video data for fine-tuning, run the following commands in order:
```{bash}
beast extract \
    --input /your/path/videos/finetune/leftCamera.video \
    --output /your/path/extracted_frames/finetune/leftCamera.video \
    --method pca_kmeans \
    --frames-per-video 700

beast extract \
    --input /your/path/videos/finetune/rightCamera.video \
    --output /your/path/extracted_frames/finetune/rightCamera.video \
    --method precomputed \
    --frames-per-video 700
```

#### Extract video frames for model pre-training

(< 1.5 hours on CPU) To extract video data for pre-training, run the following commands in order:
```{bash}
beast extract \
    --input /your/path/videos/pretrain/leftCamera.video \
    --output /your/path/extracted_frames/pretrain/leftCamera.video \
    --method pca_kmeans \
    --frames-per-video 400

beast extract \
    --input /your/path/videos/pretrain/rightCamera.video \
    --output /your/path/extracted_frames/pretrain/rightCamera.video \
    --method precomputed \
    --frames-per-video 400
```

#### Extract neural and behavior data for model evaluation

Run the following command to extract data for the 5 EIDs used for fine-tuning and evaluation:
```{bash}
source scripts/batch_extract_neural_data.sh ONE_CACHE_PATH VIDEO_TIMESTAMPS OUTPUT_PATH NUM_TRIALS
```
**Arguments**:
- `ONE_CACHE_PATH`
   Path to the directory where raw data downloaded from the IBL database will be cached.
- `VIDEO_TIMESTAMPS`
   Path to the directory containing video timestamp files.
   *(Note: left and right videos share the same timestamps.)*
- `OUTPUT_PATH`
   Directory where the extracted neural and behavioral data will be saved.
- `NUM_TRIALS`
   Number of time intervals to extract from the neural data.
   *(Note: these intervals do not follow the trial structure.)*


#### Extract video frames using the extracted timestamps

TODO
