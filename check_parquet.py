import pandas as pd
import json

# 读取一个数据文件
df = pd.read_parquet("/mnt/workspace/datasets/RoboTwin2.0/dataset_lerobot/aloha-agilex_clean_50/data/chunk-000/episode_000000.parquet")

# 查看所有列名
print("数据文件中的列名：")
print(df.columns.tolist())

# 查看是否有 annotation 相关的列
annotation_cols = [col for col in df.columns if 'annotation' in col.lower() or 'human' in col.lower()]
print("\nAnnotation 相关列：")
print(annotation_cols)