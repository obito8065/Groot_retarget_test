#!/bin/bash
# 测评GR00t-n15的对齐动作空间 for robocasa:
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA

# 设置渲染环境变量（用于headless环境）
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 scripts/batch_robocasa_eval.py \
    --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robocasa_ckpt_24tasks_sample/n1.5_nopretrain_finetune_on_robocasa_eepose_v0.1/checkpoint-24000 \
    --embodiment_tag robocasa \
    --data_config fourier_gr1_arms_waist \
    --env_names \
        gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env \
    --base_port 9220 \
    --max_episode_steps 720 \
    --n_envs 5 \
    --n_episodes 100 \
    --use_eepose \
    --base_video_dir /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_video_record/output_robocasa_24tasks_sample/n1.5_nopretrain_finetune_on_robocasa_ee_v0.1/24k