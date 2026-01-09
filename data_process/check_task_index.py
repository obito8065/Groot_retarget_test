import os
import json
from pathlib import Path

# 数据集根目录
DATASET_ROOT = "/home/robot/pi/datasets/rm_lerobot"
DAY_LIST = [f"day{i}" for i in range(1, 11)]

for day in DAY_LIST:
    jsonl_path = Path(DATASET_ROOT) / day / "meta" / "new_tasks.jsonl"
    print(f"\n检查 {jsonl_path} ...")
    if not jsonl_path.exists():
        print("  文件不存在！")
        continue
    task_indices = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                task_indices.append(obj["task_index"])
            except Exception as e:
                print(f"  解析出错: {e}")
    if not task_indices:
        print("  没有找到任何 task_index！")
        continue
    task_indices = sorted(task_indices)
    missing = []
    for idx in range(task_indices[0], task_indices[-1] + 1):
        if idx not in task_indices:
            missing.append(idx)
    if missing:
        print(f"  缺失的 task_index: {missing}")
    else:
        print("  task_index 连续，无缺失。") 