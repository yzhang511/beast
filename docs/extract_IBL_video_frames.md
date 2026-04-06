## Extract IBL video frames


#### Extract data for fine-tuning

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

#### Extract data for pre-training

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

#### Extract data for evaluation

TODO
