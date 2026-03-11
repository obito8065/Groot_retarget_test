#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /vla/users/lijiayi/unifytip_groot

SEED=0
export PYTHONHASHSEED=$SEED
export PYTHONUNBUFFERED=1


# conda activate robocasa

export CUDA_VISIBLE_DEVICES=3
python3 scripts/simulation_service.py \
        --client \
        --env_name gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        --port 57309 \
        --host localhost \
        --video_dir /vla/users/lijiayi/unifytip_groot/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_retarget_v5_1000ep_bs384_horizon50_taskR_lr1e5/50ksteps-batch0-100episodes \
         --n_episodes 100 \
        --n_envs 1 \
        --max_episode_steps 500 \
        --n_action_steps 50 \
        --episode_seed_start 0

