import json
from collections import OrderedDict

DATA_PATH = "/vla/users/zhaolin/datasets/rm_lerobot_merge"

EPISODES_PATH = f"{DATA_PATH}/meta/episodes.jsonl"
TASKS_PATH = f"{DATA_PATH}/meta/new_tasks.jsonl"

def main():
    tasks_list = []
    with open(EPISODES_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            task = data["tasks"]
            tasks_list.append(task)
    # 重写 tasks.jsonl
    with open(TASKS_PATH, 'w', encoding='utf-8') as f:
        for idx, task in enumerate(tasks_list):
            f.write(json.dumps({"task_index": idx, "task": task}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
