import os
import re
import sys
import time
import logging
import argparse
import subprocess
import shlex
import signal
import socket
from datetime import datetime
from multiprocessing import Process, Queue
from typing import List, Set
import random
import numpy as np
from set_log import setup_logging
# 固定随机种子
def set_seed_everywhere(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def is_port_available(port: int, host: str = 'localhost') -> bool:
    """检查端口是否可用"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            return s.connect_ex((host, port)) != 0
    except:
        return False
        
def find_available_port(start_port: int, used_ports: set, host: str = 'localhost') -> int:
    """从起始端口开始寻找未被占用的端口"""
    port = start_port
    while port in used_ports or not is_port_available(port, host):
        port += 1
    return port

def allocate_ports_for_tasks(num_tasks: int, base_port: int, host: str = 'localhost', logger = None) -> List[int]:
    """为所有任务预先分配不冲突的端口"""
    allocated_ports = []
    used_ports = set()
    if logger:
        logger.info(f"\n{'='*80}\n检查并分配可用端口...\n{'='*80}")
    for task_id in range(num_tasks):
        preferred_port = base_port + task_id
        allocated_port = find_available_port(preferred_port, used_ports, host)
        allocated_ports.append(allocated_port)
        used_ports.add(allocated_port)
        if logger and allocated_port != preferred_port:
            logger.warning(f"任务 {task_id + 1}: 端口 {preferred_port} 占用，改用 {allocated_port}")
    return allocated_ports


def run_server(
    repo_root: str, 
    model_path: str, 
    data_config: str, 
    embodiment_tag: str, 
    port: int, 
    gpu_id: int, 
    use_eepose: bool, 
    log_path: str,
    queue,
    task_name: str,
    ):
    """启动 GR00T 推理服务端"""

    # 获取 logger 并配置输出到该任务的特定日志文件
    logger = logging.getLogger(f"{task_name}_Server")
    logger.setLevel(logging.INFO)
    # 使用与 set_log.py 一致的格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 构建推理服务命令
        cmd = [
            "python3", "scripts/inference_service.py",
            "--server",
            "--model_path", model_path,
            "--data_config", data_config,
            "--embodiment_tag", embodiment_tag,
            "--port", str(port)
        ]
        if use_eepose:
            cmd.append("--use_eepose")

        logger.info(f"[{task_name}] 启动 Server: GPU {gpu_id}, Port {port}")
        logger.info(f"[{task_name}] 命令: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, cwd=repo_root, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        # 实时将 Server 输出重定向到任务日志文件
        for line in process.stdout:
            logger.info(f"[Server] {line.rstrip()}")
        process.wait()
        queue.put(('server', task_name, port, process.returncode))
    except Exception as e:
        logger.info(f"[{task_name}] Server 异常: {e}")
        queue.put(('server', task_name, port, -1))


def run_client(
    robotwin_root: str,
    task_name: str, 
    task_config: str, 
    port: int, 
    seed: int, 
    save_dir: str, 
    n_episodes: int, 
    gpu_id: int, 
    log_path: str, 
    queue
    ):
    """启动 RoboTwin 仿真客户端"""
    # 获取 logger 并配置输出到该任务的特定日志文件
    logger = logging.getLogger(f"{task_name}_Client")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # 注入 RoboTwin 必要的环境变量
        env["LD_LIBRARY_PATH"] = "/tmp/nvidia-gl-extract/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
        env["VK_ICD_FILENAMES"] = "/usr/share/vulkan/icd.d/nvidia_icd.json"

        # 使用 bash 执行以确保环境激活
        
        cmd_str = (
            f"cd {robotwin_root} && "
            f"source /mnt/workspace/envs/conda3/bin/activate robotwin2_n && "
            f"python3 -u script/groot_simulation_client.py "
            f"--task_name {task_name} --task_config {task_config} --seed {seed} "
            f"--host localhost --port {port} --save_dir {os.path.join(save_dir, task_name)} "
            f"--test_num {n_episodes}"
        )
        logger.info(f"[{task_name}] 启动 Client: GPU {gpu_id}, Port {port}")
        # logger.info(f"[{task_name}] 命令: {' '.join(cmd_str)}")

        process = subprocess.Popen(
                ['bash', '-c', cmd_str], cwd=robotwin_root, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
            )

        success_rate = 0.0
        for line in process.stdout:
            logger.info(f"[Client] {line.rstrip()}")
            # 实时从输出中提取成功率
            if 'Success rate:' in line:
                match = re.search(r'Success rate:\s*([\d.]+)', line)
                if match:
                    success_rate = float(match.group(1))
        
        process.wait()
        queue.put(('client', task_name, port, process.returncode, success_rate))
    except Exception as e:
        logger.info(f"[{task_name}] Client 异常: {e}")
        queue.put(('client', task_name, port, -1, None))



def run_task_pair(task_id, task_name, args, gpu_id, task_log, result_queue, port, logger):
    """管理一对 Server 和 Client 的开启与关闭"""
    task_queue = Queue()
    seed = args.seed_base + task_id
    
    # 1. 启动 Server
    logger.info(f"run_task_pair: running server for {task_name}")
    server_proc = Process(target=run_server, args=(
        args.policy_root, args.model_path, args.data_config, 
        args.embodiment_tag, port, gpu_id, args.use_eepose, 
        task_log, task_queue, task_name
    ))
    server_proc.start()
    
    # 2. 等待 Server 启动（初始化模型显存）
    time.sleep(15) 
    
    # 3. 启动 Client
    seed = args.seed_base + task_id
    logger.info(f"run_task_pair: running client for {task_name}")
    client_proc = Process(target=run_client, args=(
        args.robotwin_root, task_name, args.task_config, port, 
        seed, args.save_dir, args.n_episodes, gpu_id, task_log, task_queue
    ))
    client_proc.start()
    
    # 4. 等待 Client 结束
    client_proc.join()
    
    # 5. Client 结束后强制关闭 Server
    server_proc.terminate()
    server_proc.join(timeout=5)
    if server_proc.is_alive():
        server_proc.kill()
        
    # 6. 整理结果并上报主进程
    results = {'task_id': task_id, 'task_name': task_name, 'port': port, 'gpu_id': gpu_id}
    while not task_queue.empty():
        msg = task_queue.get()
        if msg[0] == 'client':
            results['returncode'] = msg[3]
            results['success_rate'] = msg[4]
    result_queue.put(results)

def main():
    parser = argparse.ArgumentParser(description='RoboTwin 批量集成评测脚本')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_config', type=str, required=True,
                       help='数据配置名称')
    parser.add_argument('--embodiment_tag', type=str, required=True,
                       help='embodiment tag')
    parser.add_argument('--task_names', type=str, nargs='+', required=True)
    parser.add_argument('--task_config', type=str, default="demo_clean")
    parser.add_argument('--base_port', type=int, default=8800,
                       help='起始端口号（每个任务会递增）')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='video和log的保存目录')
    parser.add_argument('--n_episodes', type=int, default=30,
                       help='每个任务的episode数量')

    parser.add_argument('--use_eepose', action='store_true',
                       help='是否使用eepose')


    # 默认路径配置：
    parser.add_argument('--policy_root', type=str, default="/mnt/workspace/users/lijiayi/GR00T_QwenVLA")
    parser.add_argument('--robotwin_root', type=str, default="/mnt/workspace/users/lijiayi/RoboTwin_utils/RoboTwin")
    parser.add_argument('--tasks_per_gpu', type=int, default=2)
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=[0,1,2,3,4,5,6,7])
    parser.add_argument('--seed_base', type=int, default=0)

    args = parser.parse_args()


    logger, main_log_path = setup_logging(args.save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_log_dir = os.path.join(args.save_dir, 'logs', f'log_{timestamp}')
    os.makedirs(task_log_dir, exist_ok=True)

    logger.info(f"{'='*80}")
    logger.info("RoboTwin 批量集成评测启动")
    logger.info(f"{'='*80}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"任务列表: {args.task_names}")
    logger.info(f"主日志文件: {main_log_path}")
    logger.info(f"子任务日志目录: {task_log_dir}")
    logger.info(f"{'='*80}\n")
    
    used_ports = set()
    result_queue = Queue()
    processes = []


    # 端口预检查并运行任务
    for i, name in enumerate(args.task_names):
        port = find_available_port(args.base_port + i, used_ports)
        used_ports.add(port)
        
        gpu_id = args.gpu_ids[(i // args.tasks_per_gpu) % len(args.gpu_ids)]
        task_log_path = os.path.join(task_log_dir, f"log_{name}.log")
        
        logger.info(f"正在启动任务 [{i+1}/{len(args.task_names)}]: {name},  分配 GPU: {gpu_id}, Port: {port}")

        # 依次运行（如果需要全并行，可将此处改为多进程 Pool）
        p = Process(target=run_task_pair, args=(
            i, name, args, gpu_id, task_log_path, result_queue, port, logger
        ))
        p.start()
        processes.append(p)
        time.sleep(2) # 错开启动


    for p in processes:
        p.join()

    # 汇总结果
    all_results = []
    while not result_queue.empty():
        all_results.append(result_queue.get())
    
    logger.info(f"\n{'='*80}")
    logger.info(f"{'任务名称':<30} | {'成功率':>7} | {'状态':<10} | {'GPU':<5}")
    logger.info(f"{'='*80}")
    all_results.sort(key=lambda x: x['task_id'])
    
    for res in all_results:
        success_rate = res.get('success_rate', 0.0)
        status = "成功" if res.get('returncode') == 0 else "运行失败"
        logger.info(f"任务: {res['task_name']:<30} | 成功率: {success_rate:>7.2%} | 状态: {status}")
    
    logger.info(f"{'='*80}\n")
    logger.info(f"详细日志目录: {task_log_dir}")

if __name__ == '__main__':
    main()