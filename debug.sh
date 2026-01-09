CUR_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH=$CUR_DIR:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export DEFAULT_EAGLE_PATH=../Qwen2.5-VL-3B-Instruct
source /vla/miniconda3/bin/activate groot

dataset_list=(
  "../libero_lerobot/libero_10_no_noops"
  "../libero_lerobot/libero_10_no_noops"
)
data_config=(
  "libero_image_wrist"
  "libero_image_wrist"
)
echo "Dataset list: ${dataset_list[@]}, ${#dataset_list[@]}"
echo "Data config: ${data_config[@]}, ${#data_config[@]}"


OUTPUT_DIR=outputs/debug
if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"

python scripts/gr00t_finetune.py \
    --master_port 12345 \
    --dataset-path ${dataset_list[@]} \
    --num-gpus 2 \
    --output_dir ${OUTPUT_DIR} \
    --video-backend torchvision_av \
    --batch_size 4 \
    --max_steps 100 \
    --save_steps 10 \
    --base_model_path ../GR00T-N1.5-3B \
    --report_to tensorboard \
    --data-config "${data_config[@]}" \
    --select_layer 12 \
    --update_backbone \
    --backbone_type qwen2_5_vl \
    --backbone_model_name_or_path ../Qwen2.5-VL-3B-Instruct \
    2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"


# --tune_llm：训练LLM
# --tune_visual：训练ViT
# --update_backbone / --backbone_type / --backbone_model_name_or_path / --select_layer：更换VLM backbone，抽取LLM第n层特征
# --update_action_head：随机初始化action head参数
# --update_action_head / --dit_num_layers：指定DiT层数，默认16
