#!/bin/bash

# source /vla/miniconda3/bin/activate groot

set -x

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

# 环境变量控制载入Qwen2.5VL模型
# export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct

DEFAULT_PORT=8000
# 检查是否提供了参数
if [ -z "$1" ]; then
  PORT=$DEFAULT_PORT
else
  PORT=$1
fi
echo "Using port: $PORT"


DEFAULT_GPU=0
# 检查是否提供了参数
if [ -z "$2" ]; then
  GPU=$DEFAULT_GPU
else
  GPU=$2
fi
echo "Using GPU: $GPU"

export CUDA_VISIBLE_DEVICES=$GPU

DEFAULT_MODEL_PATH=""
# 检查是否提供了参数
if [ -z "$3" ]; then
  MODEL_PATH=$DEFAULT_MODEL_PATH
else
  MODEL_PATH=$3
fi
echo "Using MODEL_PATH: $MODEL_PATH"

python ../scripts/groot_eval_libero.py \
--server \
--model_path $MODEL_PATH \
--embodiment-tag new_embodiment \
--data-config libero_image_wrist \
--denoising-steps 4 \
--port $PORT

