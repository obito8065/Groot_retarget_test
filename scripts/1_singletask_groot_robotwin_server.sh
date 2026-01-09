#!/bin/bash
# set -euo pipefail
# source /mnt/workspace/envs/conda3/bin/activate gr00t_p

# 10task_eepose:""   --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robotwin_ckpt_10tasks_sample/n1.5_nopretrain_finetuneALL_on_robotwin_eepose_v0.1/checkpoint-7740 \


cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA
python3 scripts/inference_service.py --server \
    --data_config robotwin_ego\
    --embodiment_tag robotwin \
    --port 7035 \
    --use_eepose \
    --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robotwin_easy_ckpt_3tasks/n1.5_nopretrain_finetuneALL_on_robotwin_eepose_v0.2/checkpoint-6390

# 