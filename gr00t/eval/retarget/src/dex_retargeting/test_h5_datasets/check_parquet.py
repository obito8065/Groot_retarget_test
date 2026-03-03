import pandas as pd

# 修改为你的 parquet 文件路径
parquet_path = "/vla/users/zhaolin/datasets/egodex_10000_lerobot_dataset/data/chunk-000/episode_000907.parquet"

# 读取 parquet 文件
df = pd.read_parquet(parquet_path)

# 打印前几个 task_index 的值
print(df["task_index"].head())