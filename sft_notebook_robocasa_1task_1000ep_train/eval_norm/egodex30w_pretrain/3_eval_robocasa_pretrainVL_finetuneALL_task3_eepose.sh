#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA

# 设置渲染环境变量（用于headless环境）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

export CUDA_VISIBLE_DEVICES=5

python3 scripts/batch_robocasa_eval.py \
    --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robocasa_ckpt_1tasks_1000ep/n1.5_pretrainVL_finetuneALL_on_robocasa_task3_v0.1/checkpoint-20000 \
    --embodiment_tag robocasa \
    --data_config fourier_gr1_arms_waist2egodex \
    --env_names \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
    --base_port 58000 \
    --max_episode_steps 720 \
    --n_envs 1 \
    --n_episodes 100 \
    --use_eepose \
    --base_video_dir /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_video_record/output_robocasa2egodex_1tasks_1000ep/task3/n1.5_pretrainVL_finetuneALL_ResumeActionhead_on_robocasa_eepose/2wsteps