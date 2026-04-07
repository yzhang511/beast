## Train model on IBL videos

#### Train from scratch

To train a single-session model on a video from each session, run the following command:
```{bash}
beast train --config configs/vit.yaml \
  --data /your/data/path \
  --output /your/checkpoint/path
```
NOTE: 
- We use a ViT model as an example, but you can change that to a ResNet auto-encoder if needed.
- Remember to train on video data from each camera view (left and right).

#### Pretrain

**NOTE**: For each view, please figure out how to pretrain a model on video frames extracted from multiple sessions on your own.

#### Fine-tune

If you want to fine-tune a pretrained model on data from a new session, you need to update the following checkpoint directory in `configs/vit.yaml` to point to your pretrained model checkpoint:
```
model:
  checkpoint: null  # load weights from checkpoint
```
Then run the same command:
```
beast train --config configs/vit.yaml \
  --data /your/data/path \
  --output /your/checkpoint/path
```
