#!/bin/bash
# 杀掉所有 batch_robocasa_eval.py 相关的进程

echo "=========================================="
echo "正在查找并清理评测进程..."
echo "=========================================="

# 查找所有相关的进程
MAIN_PIDS=$(pgrep -f "batch_robocasa_eval.py")
SERVER_PIDS=$(pgrep -f "inference_service.py.*--server")
CLIENT_PIDS=$(pgrep -f "simulation_service.py.*--client")

# 统计进程数量
main_count=$(echo "$MAIN_PIDS" | grep -v "^$" | wc -l)
server_count=$(echo "$SERVER_PIDS" | grep -v "^$" | wc -l)
client_count=$(echo "$CLIENT_PIDS" | grep -v "^$" | wc -l)

echo "找到进程："
echo "  主进程 (batch_robocasa_eval.py): $main_count 个"
echo "  服务端进程 (inference_service.py): $server_count 个"
echo "  客户端进程 (simulation_service.py): $client_count 个"
echo ""

# 杀掉主进程及其所有子进程
if [ ! -z "$MAIN_PIDS" ]; then
    echo "正在清理主进程..."
    for pid in $MAIN_PIDS; do
        if [ ! -z "$pid" ]; then
            echo "  杀掉主进程 $pid 及其所有子进程..."
            # 先杀掉所有子进程
            pkill -P $pid 2>/dev/null
            # 等待一下让子进程退出
            sleep 1
            # 杀掉主进程
            kill $pid 2>/dev/null
        fi
    done
    sleep 2
fi

# 杀掉所有服务端进程
if [ ! -z "$SERVER_PIDS" ]; then
    echo "正在清理服务端进程..."
    for pid in $SERVER_PIDS; do
        if [ ! -z "$pid" ]; then
            echo "  杀掉服务端进程 $pid..."
            kill $pid 2>/dev/null
        fi
    done
    sleep 1
fi

# 杀掉所有客户端进程
if [ ! -z "$CLIENT_PIDS" ]; then
    echo "正在清理客户端进程..."
    for pid in $CLIENT_PIDS; do
        if [ ! -z "$pid" ]; then
            echo "  杀掉客户端进程 $pid..."
            kill $pid 2>/dev/null
        fi
    done
    sleep 1
fi

# 再次检查，强制杀掉仍在运行的进程
echo ""
echo "检查残留进程，强制清理..."

# 强制杀掉所有相关进程
pkill -9 -f "batch_robocasa_eval.py" 2>/dev/null
pkill -9 -f "inference_service.py.*--server" 2>/dev/null
pkill -9 -f "simulation_service.py.*--client" 2>/dev/null

# 等待进程退出
sleep 2

# 最终检查
REMAINING_MAIN=$(pgrep -f "batch_robocasa_eval.py" | wc -l)
REMAINING_SERVER=$(pgrep -f "inference_service.py.*--server" | wc -l)
REMAINING_CLIENT=$(pgrep -f "simulation_service.py.*--client" | wc -l)

echo ""
echo "=========================================="
echo "清理完成！"
echo "=========================================="
echo "剩余进程："
echo "  主进程: $REMAINING_MAIN 个"
echo "  服务端进程: $REMAINING_SERVER 个"
echo "  客户端进程: $REMAINING_CLIENT 个"

if [ "$REMAINING_MAIN" -eq 0 ] && [ "$REMAINING_SERVER" -eq 0 ] && [ "$REMAINING_CLIENT" -eq 0 ]; then
    echo ""
    echo "✓ 所有进程已成功清理！"
    exit 0
else
    echo ""
    echo "⚠ 仍有进程残留，请手动检查："
    if [ "$REMAINING_MAIN" -gt 0 ]; then
        echo "  主进程 PID:"
        pgrep -f "batch_robocasa_eval.py" | sed 's/^/    /'
    fi
    if [ "$REMAINING_SERVER" -gt 0 ]; then
        echo "  服务端进程 PID:"
        pgrep -f "inference_service.py.*--server" | sed 's/^/    /'
    fi
    if [ "$REMAINING_CLIENT" -gt 0 ]; then
        echo "  客户端进程 PID:"
        pgrep -f "simulation_service.py.*--client" | sed 's/^/    /'
    fi
    exit 1
fi