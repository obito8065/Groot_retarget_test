#!/bin/bash
# 仅清理 RobotWin eval 相关进程，避免误杀 robocasa 的 server

echo "=========================================="
echo "正在查找并清理 RobotWin 评测进程..."
echo "=========================================="

# 1) 主进程：batch_robotwin_eval.py
MAIN_PIDS=$(pgrep -f "batch_robotwin_eval.py")

# 2) 仅匹配 RobotWin 的 GR00T 推理服务端（关键：限定 data_config/embodiment_tag）
SERVER_PIDS=$(
  pgrep -f "inference_service.py.*--server.*--data_config[[:space:]]+robotwin_ego" || true
)
# 兜底：有的命令可能没带 data_config，但带 embodiment_tag（再加一条 OR）
SERVER_PIDS_2=$(
  pgrep -f "inference_service.py.*--server.*--embodiment_tag[[:space:]]+robotwin" || true
)

# 合并去重
SERVER_PIDS=$(echo -e "$SERVER_PIDS\n$SERVER_PIDS_2" | awk 'NF' | sort -u)

# 3) RoboTwin 仿真客户端
CLIENT_PIDS=$(pgrep -f "groot_simulation_client.py" || true)

# 统计进程数量
main_count=$(echo "$MAIN_PIDS" | awk 'NF' | wc -l)
server_count=$(echo "$SERVER_PIDS" | awk 'NF' | wc -l)
client_count=$(echo "$CLIENT_PIDS" | awk 'NF' | wc -l)

echo "找到进程："
echo "  主进程 (batch_robotwin_eval.py): $main_count 个"
echo "  服务端进程 (inference_service.py --server, robotwin only): $server_count 个"
echo "  客户端进程 (groot_simulation_client.py): $client_count 个"
echo ""

# 先杀掉主进程及其子进程（尽量温和退出）
if [ ! -z "$MAIN_PIDS" ]; then
  echo "正在清理主进程及其子进程..."
  for pid in $MAIN_PIDS; do
    if [ ! -z "$pid" ]; then
      echo "  终止主进程 $pid 的子进程..."
      pkill -P $pid 2>/dev/null
      sleep 1
      echo "  终止主进程 $pid..."
      kill $pid 2>/dev/null
    fi
  done
  sleep 2
fi

# 再杀 robotwin server（只杀匹配到的 PIDs）
if [ ! -z "$SERVER_PIDS" ]; then
  echo "正在清理 RobotWin 服务端进程..."
  for pid in $SERVER_PIDS; do
    if [ ! -z "$pid" ]; then
      echo "  终止服务端进程 $pid..."
      kill $pid 2>/dev/null
    fi
  done
  sleep 1
fi

# 再杀客户端
if [ ! -z "$CLIENT_PIDS" ]; then
  echo "正在清理客户端进程..."
  for pid in $CLIENT_PIDS; do
    if [ ! -z "$pid" ]; then
      echo "  终止客户端进程 $pid..."
      kill $pid 2>/dev/null
    fi
  done
  sleep 1
fi

echo ""
echo "检查残留进程，强制清理（仍然仅限 robotwin server 条件）..."
pkill -9 -f "batch_robotwin_eval.py" 2>/dev/null
pkill -9 -f "inference_service.py.*--server.*--data_config[[:space:]]+robotwin_ego" 2>/dev/null
pkill -9 -f "inference_service.py.*--server.*--embodiment_tag[[:space:]]+robotwin" 2>/dev/null
pkill -9 -f "groot_simulation_client.py" 2>/dev/null

sleep 2

# 最终检查
REMAINING_MAIN=$(pgrep -f "batch_robotwin_eval.py" | wc -l)
REMAINING_SERVER=$(
  pgrep -f "inference_service.py.*--server.*--data_config[[:space:]]+robotwin_ego" | wc -l
)
REMAINING_SERVER_2=$(
  pgrep -f "inference_service.py.*--server.*--embodiment_tag[[:space:]]+robotwin" | wc -l
)
REMAINING_CLIENT=$(pgrep -f "groot_simulation_client.py" | wc -l)

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo "剩余进程："
echo "  主进程: $REMAINING_MAIN 个"
echo "  服务端进程(robotwin): $((REMAINING_SERVER + REMAINING_SERVER_2)) 个"
echo "  客户端进程: $REMAINING_CLIENT 个"

exit 0