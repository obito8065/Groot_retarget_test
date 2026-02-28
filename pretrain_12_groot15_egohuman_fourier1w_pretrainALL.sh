#!/bin/bash
export JOB_NAME=$(hostname  | awk -F'-' '{print $1"-"$2}')
export JOB_LOGDIR=/mnt/workspace/job-logs/${JOB_NAME=$}
mkdir -p $JOB_LOGDIR

# source /vla/miniconda3/bin/activate gr00t_n15
# 无论是运行挂载盘下的代码，还是运行容器内代码，注意进入工作目录。
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 这个变量不起作用。
export NCCL_TIMEOUT=1800


RDMA_DEVICES=$(ls /sys/class/infiniband)
if [ -z "$RDMA_DEVICES" ]; then
  log "ERROR: No active RDMA devices found. Exiting script." >&2
  exit 1
fi

# 设置RDMA设备列表 (逗号分隔)
NCCL_IB_HCA=$(echo "$RDMA_DEVICES" | tr '\n' ',' | sed 's/,$//')
export NCCL_IB_HCA
echo "Detected RDMA devices: $NCCL_IB_HCA"

# 获取GID_INDEX
NCCL_IB_GID_INDEX=""
output=$(show_gids | grep v2)
# 遍历每一行
while IFS= read -r line; do
  ipv4=$(echo "$line" | awk '{print $5}')
  # 检查IPv4地址是否有值（不是空字符串且不是全0地址）
  if [[ -n "$ipv4" && "$ipv4" != "0000:0000:0000:0000:0000:ffff:0000:0000" && "$ipv4" =~ [0-9]+\.[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    NCCL_IB_GID_INDEX=$(echo "$line" | awk '{print $3}')
    # 找到第一个匹配项后立即退出循环
    break
  fi
done <<<"$output"

# 自动检测主网卡（优先 ib0, 否则 eth0）
# if ip link show ib0 &>/dev/null; then
#  NCCL_SOCKET_IFNAME="ib0"
# elif ip link show eth0 &>/dev/null; then
#  NCCL_SOCKET_IFNAME="eth0"
# else
   # 自动选第一个 up 的网卡
#  NCCL_SOCKET_IFNAME=$(ip -o link show | awk -F': ' '{print $2}' | grep -v lo | head -n1)
# fi
export NCCL_SOCKET_IFNAME="eth0"

# 处理分布式训练需要使用到的相关环境变量
export NCCL_IB_HCA=${NCCL_IB_HCA}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}
export NCCL_NET_GDR_LEVEL=2
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}
# export NCCL_DEBUG=TRACE # 调试的时候打开
# export NCCL_DEBUG_FILE=$JOB_LOGDIR/nccl-debug-$(hostname).log

set -x 
echo -e "\n############## start pretrain ###############\n"
echo -e "\n########## New Action Head ##########\n"
DEBUG=false

# ==== 多节点配置 ====
# 总节点数（根据实际情况修改）
NUM_NODES=${PET_NNODES}
# 当前节点索引（0-based，每个节点启动时需指定不同值）
NODE_RANK=${PET_NODE_RANK}
# 主节点IP地址（确保所有节点可访问）
MASTER_ADDR=${PET_MASTER_ADDR}  # 替换为实际主节点IP
# 主节点端口（保持与代码中一致）
MASTER_PORT=${PET_MASTER_PORT}

# ==== debug or normal mode switch ====
if [ "$DEBUG" = "true" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "Debug mode"
    NUM_GPUS=1
    BATCH_SIZE=96
    WORKERS=0
else
    echo "Normal mode"
    # 每个节点使用的GPU（根据节点实际GPU数量调整）
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    NUM_GPUS_PER_NODE=8  # 每个节点的GPU数量
    NUM_GPUS=$NUM_GPUS_PER_NODE  # 补充定义：单节点的GPU数（用于条件判断）
    TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))  # 总GPU数量（多节点）
    BATCH_SIZE=256  # 每个GPU的batch size
    WORKERS=8
fi


SAVE_STEPS=2000  # 2000
EPOCHS=20

SOURCE_PATH=$(pwd)
export PYTHONPATH=$SOURCE_PATH:$PYTHONPATH

# DATASET_DIR=/export1/vla/datasets/lerobot_oxe

dataset_list=(
    /mnt/workspace/datasets/lerobot_dataset
)
data_config=(
  "ego_dex_standard"
)


OUTPUT_DIR=/mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/n1.5_pretrain_egodex_fourier1w_pretrainALL_v1.0


if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

script_path="$(realpath "$0")"
cp "$script_path" "$OUTPUT_DIR"


#########################################################
# 预训练阶段: freeze VLM backbone, 微调阶段再训练ViT或LLM
#########################################################

# 多节点训练使用torchrun启动

env IS_TORCHRUN=1 torchrun \
  --nnodes=$NUM_NODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  --nproc_per_node=$NUM_GPUS_PER_NODE \
  scripts/gr00t_finetune.py \
  --dataset_path "${dataset_list[@]}" \
  --num-gpus $NUM_GPUS_PER_NODE \
  --batch-size $BATCH_SIZE \
  --output-dir $OUTPUT_DIR \
  --num-train-epochs $EPOCHS \
  --no-instr-use-episode-index \
  --video-backend torchvision_av \
  --dataloader_num_workers $WORKERS \
  --base_model_path /mnt/workspace/users/lijiayi/checkpoints/GR00T-N1.5-3B \
  --action_horizon 16 \
  --report_to tensorboard \
  --save_steps $SAVE_STEPS \
  --data-config "${data_config[@]}" \
  --tune_visual \
  --tune-llm \
  --learning_rate 3.2e-4 \
  --update_action_head 2>&1 | tee ${JOB_LOGDIR}/$(hostname).log

