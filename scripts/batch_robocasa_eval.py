# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
批量评测脚本：管理多个场景任务的并行评测

使用方法:
cd /vla/users/lijiayi/code/GR00T_QwenVLA

python3 scripts/batch_robocasa_eval.py \
    --model_path /vla/users/lijiayi/code/GR00T_QwenVLA/output_robocasa_checkpoints/mutitask-12/n1.5_nopretrain_finetune_on_robocasa_tunevl_v1.0/checkpoint-12000 \
    --embodiment_tag robocasa \
    --data_config fourier_gr1_arms_waist \
    --env_names \
        gr1_unified/PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_Env \
        gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env \
    --base_port 9400 \
    --max_episode_steps 720 \
    --n_envs 5 \
    --n_episodes 30 \
    --base_video_dir /vla/users/lijiayi/code/GR00T_QwenVLA/outputs_robocasa/multitask12_video_dir_nopretrain_12k

实现效果：会生成如下目录
    multitask12_video_dir/
    ├── videos_{task_name1}/          # 任务1的视频目录
    ├── videos_{task_name2}/          # 任务2的视频目录
    ├── videos_{task_name3}/          # 任务3的视频目录（如果有）
    └── logs/                         # 日志目录
        ├── batch_eval_{timestamp}.log  # 主日志文件（汇总信息）
        └── log_{timestamp}/            # 带时间戳的任务日志目录
            ├── log_{task_name1}.log    # 任务1的详细日志（server+client输出）
            ├── log_{task_name2}.log    # 任务2的详细日志（server+client输出）
            └── log_{task_name3}.log    # 任务3的详细日志（如果有）



"""

import os
import sys
import time
import subprocess
import multiprocessing
from multiprocessing import Process, Queue
from typing import List, Dict, Tuple
import argparse
import re
from pathlib import Path
from set_log import setup_logging
from datetime import datetime
import socket

# 固定随机种子
def set_seed_everywhere(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def is_port_available(port: int, host: str = 'localhost') -> bool:
    """检查端口是否可用（未被占用）"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex((host, port))
            # 如果连接失败（result != 0），说明端口可用
            return result != 0
    except Exception:
        return False

def find_available_port(start_port: int, used_ports: set, host: str = 'localhost') -> int:
    """找到从start_port开始的第一个可用端口，跳过已使用的端口"""
    port = start_port
    while port in used_ports or not is_port_available(port, host):
        port += 1
    return port

def allocate_ports_for_tasks(
    num_tasks: int, 
    base_port: int, 
    host: str = 'localhost',
    logger = None
) -> List[int]:
    """
    为所有任务分配可用端口
    
    Args:
        num_tasks: 任务数量
        base_port: 起始端口号
        host: 主机地址
        logger: 日志记录器（可选）
    
    Returns:
        端口号列表，每个任务对应一个端口
    """
    allocated_ports = []
    used_ports = set()
    
    if logger:
        logger.info(f"\n{'='*80}")
        logger.info("检查并分配端口...")
        logger.info(f"{'='*80}")
    
    for task_id in range(num_tasks):
        # 首选端口：base_port + task_id
        preferred_port = base_port + task_id
        
        # 检查首选端口是否可用
        if preferred_port not in used_ports and is_port_available(preferred_port, host):
            allocated_port = preferred_port
            if logger:
                logger.info(f"任务 {task_id + 1}: 使用首选端口 {allocated_port} ✓")
        else:
            # 如果首选端口不可用，查找下一个可用端口
            allocated_port = find_available_port(preferred_port, used_ports, host)
            if logger:
                if preferred_port in used_ports:
                    logger.warning(f"任务 {task_id + 1}: 首选端口 {preferred_port} 已被其他任务使用，使用端口 {allocated_port}")
                else:
                    logger.warning(f"任务 {task_id + 1}: 首选端口 {preferred_port} 被占用，使用端口 {allocated_port}")
        
        allocated_ports.append(allocated_port)
        used_ports.add(allocated_port)
    
    if logger:
        logger.info(f"{'='*80}\n")
    
    return allocated_ports

def extract_task_name(env_name: str) -> str:
    """从env_name中提取任务名称"""
    # 例如: gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env
    # 提取: PosttrainPnPNovelFromTrayToTieredbasketSplitA
    parts = env_name.split('/')
    if len(parts) > 1:
        task_part = parts[-1]
        # 移除后缀如 _GR1ArmsAndWaistFourierHands_Env
        task_part = re.sub(r'_GR1.*$', '', task_part)
        return task_part
    return env_name.split('/')[-1]


def run_server(
    model_path: str,
    embodiment_tag: str,
    data_config: str,
    port: int,
    gpu_id: int,
    gr00t_env: str,
    gr00t_dir: str,
    queue: Queue,
    task_name: str,
    log_file_path: str,
    backbone_type: str = None,
    use_eepose: bool = False,
    use_fourier_hand_retarget: bool = False,
    denoising_steps: int = 4,
    server_seed: int = 0, # 服务端policy的随机种子
):
    """在子进程中运行服务端"""
    # 设置日志文件
    log_file = open(log_file_path, 'a', encoding='utf-8')
    

    def log_print(*args, **kwargs):
        """同时输出到控制台和日志文件"""
        message = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")
        log_file.flush()


    try:
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(0)
        
        # 切换到 GR00T 目录
        os.chdir(gr00t_dir)
        
        # 构建命令
        cmd = [
            'python3', 'scripts/inference_service.py',
            '--server',
            '--model_path', model_path,
            '--embodiment_tag', embodiment_tag,
            '--data_config', data_config,
            '--port', str(port),
            '--seed', str(server_seed),  # 随机种子
        ]
        
        if backbone_type:
            cmd.extend(['--backbone_type', backbone_type])
        if use_eepose:
            cmd.append('--use_eepose')
        if use_fourier_hand_retarget:
            cmd.append('--use_fourier_hand_retarget')
        cmd.extend(['--denoising_steps', str(denoising_steps)])
        
        log_print(f"[{task_name}] 启动服务端: GPU {gpu_id}, Port {port}")
        log_print(f"[{task_name}] 命令: {' '.join(cmd)}")
        
        # 运行服务端（这会阻塞直到服务端停止）
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志
        for line in process.stdout:
            log_print(f"[{task_name} Server] {line.rstrip()}")
        
        process.wait()
        log_file.close()
        queue.put(('server', task_name, port, process.returncode))
        
    except Exception as e:
        log_print(f"[{task_name}] 服务端错误: {e}")
        log_file.close()
        queue.put(('server', task_name, port, -1))


# 运行client进程，仿真环境
def run_client(
    env_name: str,
    port: int,
    host: str,
    video_dir: str,
    n_episodes: int,
    n_envs: int,
    max_episode_steps: int,
    n_action_steps: int,
    robocasa_env: str,
    robocasa_dir: str,
    queue: Queue,
    task_name: str,
    log_file_path: str,
    gpu_id: int,
    episode_seed_start: int,  # <<< 新增仿真评测任务的每个episode的随机种子起始值
):
    """在子进程中运行客户端"""
    # 设置日志文件
    log_file = open(log_file_path, 'a', encoding='utf-8')

    def log_print(*args, **kwargs):
        """同时输出到控制台和日志文件"""
        message = ' '.join(str(arg) for arg in args)
        print(*args, **kwargs)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"[{timestamp}] {message}\n")
        log_file.flush()
        
    try:
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # 设置MuJoCo EGL设备ID（与GPU ID一致，分散负载）
        # 系统有8个EGL设备可用，每个进程使用对应的设备
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(0)

        # 切换到robocasa目录
        os.chdir(robocasa_dir)
        
        # 构建命令 - 使用bash执行source命令
        cmd_str = (
            f'source /mnt/workspace/envs/conda3/bin/activate robocasa && '
            f'PYTHONHASHSEED={episode_seed_start} PYTHONUNBUFFERED=1 '
            f'python3 scripts/simulation_service.py '
            f'--client '
            f'--env_name {env_name} '
            f'--port {port} '
            f'--host {host} '
            f'--video_dir {video_dir} '
            f'--n_episodes {n_episodes} '
            f'--n_envs {n_envs} '
            f'--max_episode_steps {max_episode_steps} '
            f'--n_action_steps {n_action_steps} '
            f'--episode_seed_start {episode_seed_start} '  # 随机种子起始值
        )
        
        log_print(f"[{task_name}] 启动客户端: GPU {gpu_id}, Port {port}")
        log_print(f"[{task_name}] 命令: {cmd_str}")
        
        # 等待服务端启动
        time.sleep(5)
        
        # 运行客户端 - 使用bash执行命令
        process = subprocess.Popen(
            ['bash', '-c', cmd_str],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时输出日志并提取成功率
        success_rate = None
        for line in process.stdout:
            log_print(f"[{task_name} Client] {line.rstrip()}")
            # 尝试从输出中提取成功率
            if 'Success rate:' in line:
                try:
                    match = re.search(r'Success rate:\s*([\d.]+)', line)
                    if match:
                        success_rate = float(match.group(1))
                except:
                    pass
        
        process.wait()
        log_file.close()
        queue.put(('client', task_name, port, process.returncode, success_rate))
        
    except Exception as e:
        log_print(f"[{task_name}] 客户端错误: {e}")
        log_file.close()
        queue.put(('client', task_name, port, -1, None))


def run_task(
    task_id: int,
    env_name: str,
    model_path: str,
    embodiment_tag: str,
    data_config: str,
    base_port: int,
    gpu_id: int,
    base_video_dir: str,
    n_episodes: int,
    n_envs: int,
    max_episode_steps: int,
    n_action_steps: int,
    gr00t_env: str,
    robocasa_env: str,
    gr00t_dir: str,
    robocasa_dir: str,
    host: str,
    log_file_path: str,
    backbone_type: str = None,
    use_eepose: bool = False,
    use_fourier_hand_retarget: bool = False,
    denoising_steps: int = 4,
    result_queue: Queue = None,
    port: int = None, # 使用鉴定过的端口
    episode_seed_start: int = 0,   # 仿真评测任务的每个episode的随机种子
    server_seed: int = 0,        # 服务端policy的随机种子
):
    """运行单个任务（服务端+客户端）"""
    task_name = extract_task_name(env_name)
    if port is None:
        port = base_port + task_id
    video_dir = os.path.join(base_video_dir, f"videos_{task_name}")
    
    # 创建视频目录
    os.makedirs(video_dir, exist_ok=True)
    
    # 在日志文件中记录任务开始信息
    with open(log_file_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] 任务 {task_id + 1}: {task_name}\n")
        f.write(f"[{timestamp}] 环境: {env_name}\n")
        f.write(f"[{timestamp}] 端口: {port}\n")
        f.write(f"[{timestamp}] GPU: {gpu_id}\n")
        f.write(f"[{timestamp}] 视频目录: {video_dir}\n")
        f.write(f"{'='*80}\n\n")


    # 创建进程间通信队列
    task_queue = Queue()
    
    # 启动服务端进程
    server_process = Process(
        target=run_server,
        args=(
            model_path,
            embodiment_tag,
            data_config,
            port,
            gpu_id,
            gr00t_env,
            gr00t_dir,
            task_queue,
            task_name,
            log_file_path,
            backbone_type,
            use_eepose,
            use_fourier_hand_retarget,
            denoising_steps,
            server_seed,
        )
    )
    
    # 启动客户端进程
    client_process = Process(
        target=run_client,
        args=(
            env_name,
            port,
            host,
            video_dir,
            n_episodes,
            n_envs,
            max_episode_steps,
            n_action_steps,
            robocasa_env,
            robocasa_dir,
            task_queue,
            task_name,
            log_file_path,
            gpu_id,
            episode_seed_start
        )
    )
    
    print(f"\n{'='*80}")
    print(f"任务 {task_id + 1}: {task_name}")
    print(f"环境: {env_name}")
    print(f"端口: {port}")
    print(f"GPU: {gpu_id}")
    print(f"视频目录: {video_dir}")
    print(f"{'='*80}\n")
    
    # 启动服务端
    server_process.start()
    
    # 等待服务端启动
    time.sleep(10)
    
    # 启动客户端
    client_process.start()
    
    # 等待客户端完成
    client_process.join()
    
    # 客户端完成后，停止服务端
    server_process.terminate()
    server_process.join(timeout=10)
    if server_process.is_alive():
        server_process.kill()
    
    # 收集结果
    results = {}
    while not task_queue.empty():
        msg = task_queue.get()
        if msg[0] == 'client':
            _, name, p, returncode, success_rate = msg
            results['success_rate'] = success_rate
            results['returncode'] = returncode
        elif msg[0] == 'server':
            _, name, p, returncode = msg
            results['server_returncode'] = returncode
    
    # 在日志文件中记录任务完成信息和成功率
    with open(log_file_path, 'a', encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'='*80}\n")
        f.write(f"[{timestamp}] 任务完成: {task_name}\n")
        if 'success_rate' in results and results['success_rate'] is not None:
            success_rate = results['success_rate']
            f.write(f"[{timestamp}] 成功率: {success_rate:.4f} ({success_rate:.2%})\n")
        else:
            f.write(f"[{timestamp}] 成功率: N/A (任务可能未完成)\n")
        f.write(f"[{timestamp}] 状态: {'成功' if results.get('returncode', -1) == 0 else '失败'}\n")
        f.write(f"{'='*80}\n\n")

    # 将结果发送到主队列
    if result_queue:
        result_queue.put({
            'task_id': task_id,
            'task_name': task_name,
            'env_name': env_name,
            'port': port,
            'gpu_id': gpu_id,
            'video_dir': video_dir,
            **results
        })




def main():
    parser = argparse.ArgumentParser(description='批量评测脚本')
    
    # 模型配置
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--embodiment_tag', type=str, required=True,
                       help='embodiment tag')
    parser.add_argument('--data_config', type=str, required=True,
                       help='数据配置名称')
    
    # 任务配置
    parser.add_argument('--env_names', type=str, nargs='+', required=True,
                       help='环境名称列表')
    
    # 网络配置
    parser.add_argument('--base_port', type=int, default=8814,
                       help='起始端口号（每个任务会递增）')
    parser.add_argument('--host', type=str, default='localhost',
                       help='服务器主机地址')
    
    # 评测配置
    parser.add_argument('--n_episodes', type=int, default=30,
                       help='每个任务的episode数量')
    parser.add_argument('--n_envs', type=int, default=5,
                       help='并行环境数量')
    parser.add_argument('--max_episode_steps', type=int, default=720,
                       help='每个episode的最大步数')
    parser.add_argument('--n_action_steps', type=int, default=16,
                       help='每个环境步的动作步数')
    
    # 输出配置
    parser.add_argument('--base_video_dir', type=str, required=True,
                       help='视频保存的基础目录')
    
    # GPU配置
    parser.add_argument('--tasks_per_gpu', type=int, default=3,
                       help='每个GPU最多运行的任务数')
    parser.add_argument('--gpu_ids', type=int, nargs='+', default=None,
                       help='可用的GPU ID列表（默认使用所有GPU）')
    
    # 环境配置
    parser.add_argument('--gr00t_env', type=str, default='gr00t_p',
                       help='GR00T conda环境名称')
    parser.add_argument('--robocasa_env', type=str, default='robocasa',
                       help='RoboCasa conda环境名称')
    parser.add_argument('--gr00t_dir', type=str, default=None,
                       help='GR00T项目目录')
    parser.add_argument('--robocasa_dir', type=str, default=None,
                       help='RoboCasa项目目录（默认使用当前目录）')
    parser.add_argument('--episode_seed_start', type=int, default=0,
                       help='仿真评测任务的每个episode的随机种子起始值')
    parser.add_argument('--server_seed', type=int, default=0,
                       help='服务端policy的随机种子')
    
    # 模型参数
    parser.add_argument('--backbone_type', type=str, default=None,
                       help='VLM backbone类型')
    parser.add_argument('--use_eepose', action='store_true', default=False,
                       help='是否使用EEPos')
    parser.add_argument('--use_fourier_hand_retarget', action='store_true', default=False,
                       help='是否使用Fourier Hand Retarget')
    parser.add_argument('--denoising_steps', type=int, default=4,
                       help='去噪步数')
    
    # 并行配置
    parser.add_argument('--max_parallel_tasks', type=int, default=None,
                       help='最大并行任务数（默认等于tasks_per_gpu * GPU数量）')
    
    args = parser.parse_args()
    
    # 设置日志记录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 添加timestamp定义
    logger, log_file = setup_logging(args.base_video_dir)
    task_log_dir = os.path.join(args.base_video_dir, 'logs', f'log_{timestamp}')
    os.makedirs(task_log_dir, exist_ok=True)
    logger.info(f"任务日志目录已创建: {task_log_dir}")


    # 确定GPU列表
    if args.gpu_ids is None:
        import torch
        num_gpus = torch.cuda.device_count()
        gpu_ids = list(range(num_gpus))
    else:
        gpu_ids = args.gpu_ids
    
    # groot和robocasa的脚本都在同一个目录下，只是环境不同，因此只需要获取当前目录
    if args.gr00t_dir is None:
        args.gr00t_dir = os.getcwd() 
    
    if args.robocasa_dir is None:
        args.robocasa_dir = os.getcwd()
    
    # 计算最大并行任务数
    max_parallel = args.max_parallel_tasks
    if max_parallel is None:
        max_parallel = args.tasks_per_gpu * len(gpu_ids)
    
    logger.info(f"\n{'='*80}")
    logger.info("批量评测配置")
    logger.info(f"{'='*80}")
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"Embodiment Tag: {args.embodiment_tag}")
    logger.info(f"数据配置: {args.data_config}")
    logger.info(f"任务数量: {len(args.env_names)}")
    logger.info(f"可用GPU: {gpu_ids}")
    logger.info(f"每个GPU任务数: {args.tasks_per_gpu}")
    logger.info(f"最大并行任务数: {max_parallel}")
    logger.info(f"起始端口: {args.base_port}")
    logger.info(f"{'='*80}\n")
    
    # 为所有任务分配可用端口
    num_tasks = len(args.env_names)
    allocated_ports = allocate_ports_for_tasks(
        num_tasks=num_tasks,
        base_port=args.base_port,
        host=args.host,
        logger=logger
    )

    # 分配GPU和端口，并为每个任务创建日志文件路径
    task_configs = []
    for i, env_name in enumerate(args.env_names):
        gpu_id = gpu_ids[(i // args.tasks_per_gpu) % len(gpu_ids)]
        task_name = extract_task_name(env_name)
        log_file_path = os.path.join(task_log_dir, f"log_{task_name}.log")
        task_configs.append({
            'task_id': i,
            'env_name': env_name,
            'gpu_id': gpu_id,
            'log_file_path': log_file_path,
            'port': allocated_ports[i],  # 添加分配的端口
        })

    # 创建结果队列
    result_queue = Queue()
    
    # 使用进程池运行任务
    processes = []
    results = []
    
    # 分批运行任务
    for batch_start in range(0, len(task_configs), max_parallel):
        batch_end = min(batch_start + max_parallel, len(task_configs))
        batch = task_configs[batch_start:batch_end]
        
        batch_msg = f"\n启动批次 {batch_start // max_parallel + 1}: 任务 {batch_start + 1}-{batch_end}"
        logger.info(batch_msg)
        
        # 启动当前批次的所有任务
        for config in batch:
            p = Process(
                target=run_task,
                args=(
                    config['task_id'],
                    config['env_name'],
                    args.model_path,
                    args.embodiment_tag,
                    args.data_config,
                    args.base_port,
                    config['gpu_id'],
                    args.base_video_dir,
                    args.n_episodes,
                    args.n_envs,
                    args.max_episode_steps,
                    args.n_action_steps,
                    args.gr00t_env,
                    args.robocasa_env,
                    args.gr00t_dir,
                    args.robocasa_dir,
                    args.host,
                    config['log_file_path'], 
                    args.backbone_type,
                    args.use_eepose,
                    args.use_fourier_hand_retarget,
                    args.denoising_steps,
                    result_queue,
                    config['port'],
                    args.episode_seed_start,
                    args.server_seed,
                )
            )
            p.start()
            processes.append(p)
        
        # 等待当前批次完成
        for p in processes:
            p.join()
        
        # 收集结果
        while not result_queue.empty():
            results.append(result_queue.get())
        
        processes.clear()
    
    # 打印最终结果
    logger.info(f"\n{'='*80}")
    logger.info("评测结果汇总")
    logger.info(f"{'='*80}")
    logger.info(f"{'任务名称':<40} {'端口':<8} {'GPU':<6} {'成功率':<10} {'状态':<10}")
    logger.info(f"{'-'*80}")
    
    # 按task_id排序
    results.sort(key=lambda x: x['task_id'])

    # 记录每个场景的成功率
    logger.info("\n" + "="*80)
    logger.info("各场景任务成功率详情")
    logger.info("="*80)
    
    for result in results:
        task_name = result['task_name']
        env_name = result['env_name']
        port = result['port']
        gpu_id = result['gpu_id']
        success_rate = result.get('success_rate', 'N/A')
        returncode = result.get('returncode', -1)
        status = '成功' if returncode == 0 else '失败'
        log_file = result.get('log_file', 'N/A')
        
        if isinstance(success_rate, float):
            success_str = f"{success_rate:.2%}"
            # 记录每个场景的成功率到日志
            logger.info(f"环境: {env_name}")
            logger.info(f"  任务名称: {task_name}")
            logger.info(f"  成功率: {success_rate:.4f} ({success_str})")
            logger.info(f"  状态: {status}")
            logger.info(f"  端口: {port}, GPU: {gpu_id}")
            logger.info(f"  视频目录: {result['video_dir']}")
            logger.info(f"  日志文件: {log_file}")
            logger.info("")
        else:
            success_str = str(success_rate)
            logger.warning(f"环境: {env_name}")
            logger.warning(f"  任务名称: {task_name}")
            logger.warning(f"  成功率: N/A (任务可能未完成)")
            logger.warning(f"  状态: {status}")
            logger.warning(f"  端口: {port}, GPU: {gpu_id}")
            logger.warning(f"  视频目录: {result['video_dir']}")
            logger.warning(f"  日志文件: {log_file}")
            logger.warning("")
        
        print(f"{task_name:<40} {port:<8} {gpu_id:<6} {success_str:<10} {status:<10}")
        print(f"  环境: {result['env_name']}")
        print(f"  视频目录: {result['video_dir']}")
        print(f"  日志文件: {log_file}")
    
    logger.info(f"{'='*80}\n")
    print(f"{'='*80}\n")
    
    logger.info(f"评测完成！")
    logger.info(f"主日志文件: {log_file}")
    logger.info(f"任务日志目录: {task_log_dir}")
    logger.info("="*80)


if __name__ == '__main__':
    main()