import os
import json
import glob

# 根目录
root_dir = '/home/robot/pi/datasets/rm_lerobot/'

# 遍历所有 day* 目录
for day_dir in glob.glob(os.path.join(root_dir, 'day*')):
    meta_dir = os.path.join(day_dir, 'meta')
    new_tasks_path = os.path.join(meta_dir, 'new_tasks.jsonl')
    tasks_path = os.path.join(meta_dir, 'tasks.jsonl')
    episodes_path = os.path.join(meta_dir, 'episodes.jsonl')

    # 检查文件是否存在
    if not (os.path.exists(new_tasks_path) and os.path.exists(tasks_path) and os.path.exists(episodes_path)):
        print(f"跳过 {day_dir}，因为有文件缺失。")
        continue

    # 读取 new_tasks.jsonl 的所有 task
    with open(new_tasks_path, 'r') as f:
        new_tasks = [json.loads(line)['task'] for line in f]

    # 替换 tasks.jsonl
    with open(tasks_path, 'r') as f:
        tasks_lines = [json.loads(line) for line in f]

    for i, line in enumerate(tasks_lines):
        if i < len(new_tasks):
            line['task'] = new_tasks[i]

    with open(tasks_path, 'w') as f:
        for line in tasks_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    # 替换 episodes.jsonl
    with open(episodes_path, 'r') as f:
        episodes_lines = [json.loads(line) for line in f]

    for i, line in enumerate(episodes_lines):
        if i < len(new_tasks):
            line['tasks'] = new_tasks[i]

    with open(episodes_path, 'w') as f:
        for line in episodes_lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

    print(f"{day_dir} 处理完成！")

print("全部处理完成！")