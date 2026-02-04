#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/code/groot_retarget

# conda activate robocasa

export CUDA_VISIBLE_DEVICES=5
python3 scripts/simulation_service.py \
        --client \
        --env_name gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        --port 57200 \
        --host localhost \
        --video_dir /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify7 \
        --n_episodes 1 \
        --n_envs 1 \
        --max_episode_steps 360 \
        --save_substep_video \
        --n_action_steps 50



 f