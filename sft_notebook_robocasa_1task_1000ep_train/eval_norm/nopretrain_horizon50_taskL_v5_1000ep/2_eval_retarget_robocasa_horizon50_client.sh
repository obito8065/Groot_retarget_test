#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/unifytip_groot

# conda activate robocasa

SEED=0
export PYTHONHASHSEED=$SEED
export PYTHONUNBUFFERED=1


export CUDA_VISIBLE_DEVICES=0
python3 scripts/simulation_service.py \
        --client \
        --env_name gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        --port 51200 \
        --host localhost \
        --video_dir /vla/users/lijiayi/unifytip_groot/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_retarget_v5_1000ep_bs512_horizon50_taskL/90ksteps-modify4 \
        --n_episodes 1 \
        --n_envs 1 \
        --max_episode_steps 450 \
        --save_substep_video \
        --n_action_steps 50 \
        --episode_seed_start 4

