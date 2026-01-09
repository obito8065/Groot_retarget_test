#!/bin/bash
## 2小时倒计时，每10秒打印一次剩余时间
#COUNTDOWN=$((5*60*60))  # 2小时，单位：秒
#while [ $COUNTDOWN -gt 0 ]; do
#    HOURS=$((COUNTDOWN/3600))
#    MINUTES=$(((COUNTDOWN%3600)/60))
#    SECONDS=$((COUNTDOWN%60))
#    printf "倒计时剩余: %02d:%02d:%02d\n" $HOURS $MINUTES $SECONDS
#    sleep 10
#    COUNTDOWN=$((COUNTDOWN-10))
#done
# export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=8

dataset_list=(
    "/export/intent/linzhao/code/datasets/day_8/day1/"
    "/export/intent/linzhao/code/datasets/day_8/day2/"
    "/export/intent/linzhao/code/datasets/day_8/day3/"
    "/export/intent/linzhao/code/datasets/day_8/day4/"
    "/export/intent/linzhao/code/datasets/day_8/day5/"
    "/export/intent/linzhao/code/datasets/day_8/day6/"
    "/export/intent/linzhao/code/datasets/day_8/day7/"
    "/export/intent/linzhao/code/datasets/day_8/day8/"
    "/export/intent/linzhao/code/datasets/day_8/take_cup_62/"
)

set -x

# 每次实验应该修改这里的output name.
OUT_NAME=../outputs/resize_img_chunk50

OUT_NAME=../outputs/resize_img_chunk32_70000_day8

# realman_multiimg: 单图进行裁剪
# 增大chunk size
python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS --batch-size 100 \
--output-dir $OUT_NAME \
--max-steps 70000 \
--data-config realman_multiimg \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path ../GR00T-N1.5-3B \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights \
--action_horizon 32 \
--save_steps 2000 # \
# --resume

# 单图不更换尺寸
# 输出结果: outputs/raw_img_instr_checkpoint-40000
#python scripts/gr00t_finetune.py \
#--dataset_path ${dataset_list[@]} \
#--num-gpus $NUM_GPUS --batch-size 100 \
#--output-dir ../outputs/multi_take \
#--max-steps 40000 \
#--data-config realman_rightarm \
#--video-backend torchvision_av \
#--dataloader_num_workers 8 \
#--base_model_path ../GR00T-N1.5-3B \
#--report_to tensorboard \
#--no-balance-trajectory-weights \
#--no-balance-dataset-weights