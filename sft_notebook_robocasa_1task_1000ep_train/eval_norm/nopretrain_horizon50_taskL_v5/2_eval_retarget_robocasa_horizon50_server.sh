#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/code/groot_retarget

# 设置渲染环境变量（用于headless环境）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=5

# Debug模式：设置为1时会在policy中记录4个txt日志文件，用于调试
# 设置为0或不设置时不会记录日志，适合批量测试任务
export DEBUG_MODE=0

python3 scripts/inference_service.py --server \
    --model_path /vla/users/lijiayi/code/groot_retarget/output_ckpt/n1.5_nopretrain_finetuneALL_on_robocasa_task1_retarget_v5_bs384_horizon50/checkpoint-33000 \
    --data_config robocasa_retarget_50_horizon \
    --embodiment_tag robocasa \
    --port 57200 \
    --use_eepose \
    --use_fourier_hand_retarget \
    --action_horizon 50
