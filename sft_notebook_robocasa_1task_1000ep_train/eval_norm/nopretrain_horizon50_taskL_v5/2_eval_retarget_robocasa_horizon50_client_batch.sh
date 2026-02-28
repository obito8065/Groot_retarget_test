#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/code/groot_retarget

# conda activate robocasa

# 设置渲染环境变量（用于headless环境）

export CUDA_VISIBLE_DEVICES=6


python3 scripts/simulation_service.py \
        --client \
        --env_name gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        --port 57307 \
        --host localhost \
        --video_dir /vla/users/lijiayi/unifytip_groot/output_video_record/output_retarget_1tasks_300ep/n1.5_nopretrain_finetuneALL_on_robocasa_retarget_v5_bs384_horizon50_taskL/33ksteps-batch0 \
        --n_episodes 30 \
        --n_envs 1 \
        --max_episode_steps 600 \
        --n_action_steps 50

