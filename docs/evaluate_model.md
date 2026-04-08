## Model evaluation

**Instruction**: 

In `extract_IBL_data.md`, you can follow the instructions to extract both neural and video data that are synchronized in time.

Now, you need to figure out the following on your own:

- How to save video latents from the extracted (synchronized) video frames using your fine-tuned model?
- How to align the neural data (binned spikes) with the video latents according to the train/val/test intervals?

Then, we can perform neural encoding and decoding using baseline linear (and nonlinear) models:

- Encoding:
    - RRR: `beast/neural_encoder/rrr.py`
    - TCN: 
      - You can import the model using `from facemap.neural_prediction.neural_model import KeypointsNetwork`.
      - However, if you want to implement a similar TCN decoder, refer to the original code [here](https://github.com/MouseLand/facemap/blob/main/facemap/neural_prediction/neural_model.py#L17).
- Decoding:
    - RRR: *Please implement it yourself.*
    - TCN: *Please implement it yourself.*

NOTE:
- For encoding, the baseline model code is provided. For decoding, please follow a similar approach and implement it yourself, as you essentially only need to swap the model inputs and outputs.
- We only need to fit single-session RRR and TCN for encoding and decoding analyses.
- If you work with behavior variables (e.g., paw speed, nose speed, etc.), then you do not have video latents. In this case, the RRR and TCN model inputs will simply be the concatenated continuous behavior variables.

For both encoding and decoding, we want to perform hyperparameter tuning using `RayTune`. 

Here is some example code for training both RRR and TCN encoders with RayTune and then evaluating neural prediction quality using the bps metric:

- [train_rrr_with_tune](https://github.com/yzhang511/video-spike/blob/main/src/test.py#L445C26-L445C45)
- [train_tcn_with_tune](https://github.com/yzhang511/video-spike/blob/main/src/test.py#L447)

For decoding, you can similarly obtain the R2 metric for each behavior variable or video latent dimension, rather than a single neuron.

NOTE: 
- Since the code provided above is only an example, you can adapt it to this codebase to train the encoder and decoder with hyperparameter tuning.
- For each session, we always train on the training set, validate on the validation set, and report model performance on the test set.
- For hyperparameter tuning, you can use the default setup (e.g., the number of random models) in the provided example code.
- For neural encoding, we use bps (see [example code here](https://github.com/yzhang511/video-spike/blob/main/src/test.py#L448)) as the metric; for neural decoding, we use R2 (see [example code here](https://github.com/yzhang511/video-spike/blob/main/src/test.py#L449)) as the metric.
