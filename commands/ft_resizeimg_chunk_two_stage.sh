#/bin/bash
# export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=8

dataset_list=(
    "/export/intent/linzhao/code/datasets/day_8/day1/"
)

set -x

# 每次实验应该修改这里的output name.
OUT_NAME=../outputs/resize_img_chunk50_two_stage

# realman_multiimg: 单图进行裁剪
# 增大chunk size
python scripts/gr00t_finetune.py \
--dataset_path ${dataset_list[@]} \
--num-gpus $NUM_GPUS --batch-size 100 \
--output-dir $OUT_NAME \
--max-steps 20000 \
--data-config realman_multiimg \
--video-backend torchvision_av \
--dataloader_num_workers 8 \
--base_model_path ../outputs/resize_img_chunk50_raw/checkpoint-50000 \
--report_to tensorboard \
--no-balance-trajectory-weights \
--no-balance-dataset-weights \
--action_horizon 50 \
--save_steps 2000 

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