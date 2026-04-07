## Model evaluation

**Instruction**: 

In `extract_IBL_data.md`, you can follow the instructions to extract both neural and video data that are synchronized in time.

Now, you need to figure out the following on your own:

- How to save video latents from the extracted (synchronized) video frames using your fine-tuned model?
- How to align the neural data (binned spikes) with the video latents according to the train/val/test intervals?

Then, we can perform neural encoding and decoding using baseline linear (and nonlinear) models:

- Encoding:
    - RRR: TODO (point to where the model code is located)
    - TCN: TODO (point to where the model code is located)
- Decoding:
    - RRR
    - TCN

For encoding, the baseline model code is provided. For decoding, please follow a similar approach and implement it yourself, as you essentially only need to swap the model inputs and outputs.

**TODO**: For both encoding and decoding, we want to perform hyperparameter tuning using `RayTune`.
