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
source scripts/batch_extract_neural_data.sh ONE_CACHE_PATH VIDEO_TIMESTAMPS OUTPUT_PATH
```
**Arguments**:
- `ONE_CACHE_PATH`
   Path to the directory where raw data downloaded from the IBL database will be cached.
- `VIDEO_TIMESTAMPS`
   Path to the directory containing video timestamp files.
   *(Note: left and right videos share the same timestamps.)*
- `OUTPUT_PATH`
   Directory where the extracted neural and behavioral data will be saved.


#### Extract video frames using the extracted timestamps

(< 1 hours on CPU) To extract video frames for evaluation according to the time intervals extracted for the neural data, run the following commands for both the left and right views:
```{bash}
beast extract \
    --input /your/path/videos/finetune/leftCamera.video \
    --output /your/path/extracted_frames/eval/leftCamera.video \
    --method timestamp \
    --timestamp_dir /your/path/timestamp \
    --neural_data_dir /your/path/neural_data
```
Added a JSON mapping file (frame_index_mapping.json) written alongside the CSV in each split directory. It maps each actual PNG filename to its video frame index:

```{json}
{
  "interval0timebin0.png": 6000,
  "interval0timebin1.png": 6001,
  "interval1timebin0.png": 3000,
  ...
}
