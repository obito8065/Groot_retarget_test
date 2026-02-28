#!/bin/bash

######################################################
# run the simulator
# (libero) robot@robot:~/pi/outputs$ source ~/miniconda3/bin/activate libero
# (libero) robot@robot:~/pi/outputs$ python /home/robot/pi/env/openpi/libero_env.py --args.task-suite-name libero_10
######################################################

set -x

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
# 环境变量控制载入Qwen2.5VL模型
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct

# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_5w_step45000
# 训练修改了layers,推理时可以直接从ckpt载入,无需指定参数.

# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-45000 0.852 0.848 0.838
# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-40000
# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-50000

# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_12layers_5w_step45000
# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_5w_20000
# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w_step15000
# MODEL_PATH=../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w_step20000

# MODEL_PATH=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step45000
#MODEL_PATH=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step50000/raw

#MODEL_PATH=/home/robot/pi/outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_125000_tag_5w_step50000

# MODEL_PATH=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step50000/raw

MODEL_PATH=../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_step50000

DEFAULT_PORT=8000
# 检查是否提供了参数
if [ -z "$1" ]; then
  PORT=$DEFAULT_PORT
else
  PORT=$1
fi
echo "Using port: $PORT"

python scripts/groot_eval_libero.py \
--server \
--model_path $MODEL_PATH \
--embodiment-tag new_embodiment \
--data-config libero_image_wrist \
--denoising-steps 4 \
--backbone-type qwen2_5_vl \
--use-time-aware-action-head \
--port $PORT

