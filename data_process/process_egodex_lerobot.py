import json


def update_tasks(episodes_file, tasks_file, output_file):
    """
    读取episodes.jsonl和tasks.jsonl文件，将episodes中的tasks值更新到tasks文件中
    并保存为新文件

    参数:
        episodes_file: episodes.jsonl文件路径
        tasks_file: tasks.jsonl文件路径
        output_file: 输出文件路径
    """
    try:
        # 读取episodes.jsonl文件，提取tasks信息
        episodes_tasks = []
        with open(episodes_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 解析每一行的JSON数据
                episode = json.loads(line.strip())
                # 提取tasks值并添加到列表
                episodes_tasks.append(episode['tasks'])

        # 读取tasks.jsonl文件并更新task值
        updated_tasks = []
        with open(tasks_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                task = json.loads(line.strip())
                # 检查是否有对应的episode任务
                if idx < len(episodes_tasks):
                    task['task'] = episodes_tasks[idx]
                updated_tasks.append(task)

        # 保存更新后的内容到新文件
        with open(output_file, 'w', encoding='utf-8') as f:
            for task in updated_tasks:
                # 每行写入一个JSON对象
                f.write(json.dumps(task, ensure_ascii=False) + '\n')

        print(f"成功更新任务数据，已保存到 {output_file}")
        print(f"共处理 {len(episodes_tasks)} 个episode任务")
        print(f"共更新 {len(updated_tasks)} 个task条目")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e.filename}")
    except KeyError as e:
        print(f"错误：JSON数据中缺少必要的键 - {e}")
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")


if __name__ == "__main__":
    # 输入文件路径
    data_dir = "/vla/users/zhaolin/datasets/egodex_10000_lerobot_dataset/meta/"
    episodes_path = data_dir + "episodes.jsonl"
    tasks_path = data_dir + "tasks.jsonl"
    # 输出文件路径
    output_path = data_dir + "updated_tasks.jsonl"

    # 调用函数执行更新操作
    update_tasks(episodes_path, tasks_path, output_path)
