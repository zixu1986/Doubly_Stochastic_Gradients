#!/bin/bash

data_path="/nv/hcoc1/bxie33/scratch/imagenet/new_train_batches"
save_path="/nv/hcoc1/bxie33/scratch/imagenet/convnet_checkpoints"
test_range="801-832"
train_range="1-800"
layer_def="./example-layers/layers-imagenet-alex-cropped-rand.cfg"
layer_params="./example-layers/layer-params-imagenet-alex-cropped-rand.cfg"
data_provider="imagenet-cropped"
test_freq="20"
gpu="0"
mini="128"
#mini="256"
crop_size=16
epochs=20

python convnet.py \
  --data-path=$data_path \
  --save-path=$save_path \
  --test-range=$test_range --train-range=$train_range \
  --layer-def=$layer_def \
  --layer-params=$layer_params \
  --data-provider=$data_provider \
  --test-freq=$test_freq \
  --gpu=$gpu \
  --mini=$mini \
  --crop-border=$crop_size \
  --epochs=$epochs \
  --run-id=0
