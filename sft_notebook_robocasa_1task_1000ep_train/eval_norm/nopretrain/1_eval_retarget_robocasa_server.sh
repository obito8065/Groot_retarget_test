#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/code/groot_retarget

# 设置渲染环境变量（用于headless环境）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=5

python3 scripts/inference_service.py --server \
    --model_path /vla/users/lijiayi/code/groot_retarget/output_ckpt/n1.5_nopretrain_finetuneALL_on_robocasa_task1_retarget_v3/checkpoint-54000 \
    --data_config robocasa_retarget \
    --embodiment_tag robocasa \
    --port 57200 \
    --use_eepose \
    --use_fourier_hand_retarget