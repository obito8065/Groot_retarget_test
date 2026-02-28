#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/code/groot_retarget

# conda activate robocasa

# headless 渲染（无视频时也需要，用于观测图像）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=5
python3 scripts/simulation_service.py \
        --client \
        --env_name gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        --port 57200 \
        --host localhost \
        --video_dir /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50_task2/12ksteps-modify0 \
         --n_episodes 10 \
        --n_envs 1 \
        --max_episode_steps 420 \
        --n_action_steps 50

