import os
from pathlib import Path
from datetime import datetime
import numpy as np

def record_action_output(
    unnormalized_action, 
    save_dir, 
    save_chunk_data=True, 
    episode_id=None, 
    step_id=None, 
    batch_idx=0):
    """
    Args:
        unnormalized_action: 动作字典
        save_dir: 保存目录
        save_chunk_data: 是否保存
        episode_id: episode ID（可选，用于文件名）
        step_id: step ID（可选，用于文件名）
        batch_idx: batch索引（默认0）
    """
    if save_chunk_data and save_dir:
        # 按照 data_config 中的顺序拼接
        action_keys_order = [
            "action.right_arm",
            "action.right_hand", 
            "action.left_arm",
            "action.left_hand"
        ]
        
        available_keys = [k for k in action_keys_order if k in unnormalized_action]
        
        if available_keys:
            # 获取维度信息
            first_key = available_keys[0]
            batch_size, horizon, _ = unnormalized_action[first_key].shape
            
            # 创建保存目录
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # 为每个batch创建文件（每个step的所有chunk追加到同一个文件）
            for batch_idx in range(batch_size):
                # 生成文件名：包含episode和batch信息
                if episode_id is not None:
                    filename = save_path / f"episode_{episode_id}_batch_{batch_idx}.txt"
                else:
                    # 如果没有episode_id，使用时间戳
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = save_path / f"batch_{batch_idx}_{timestamp}.txt"
                
                # 判断文件是否存在，如果不存在则写入头部信息
                file_exists = filename.exists()
                
                # 使用追加模式打开文件
                with open(filename, 'a') as f:  # 使用 'a' 追加模式
                    # 如果是新文件，写入头部信息
                    if not file_exists:
                        dims_info = []
                        for key in available_keys:
                            dims_info.append(f"{key}:{unnormalized_action[key].shape[2]}")
                        f.write(f"# Episode: {episode_id}, Batch: {batch_idx}\n")
                        f.write(f"# Dimensions: {', '.join(dims_info)}\n")
                        f.write(f"# Total dimension per chunk: {sum(unnormalized_action[k].shape[2] for k in available_keys)}\n")
                        f.write("# Format: right_arm right_hand left_arm left_hand\n")
                        f.write("# Each line represents one chunk (16 chunks per step)\n")
                        f.write("# ==========================================\n")
                    
                    # 写入当前step的标记
                    if step_id is not None:
                        f.write(f"\n# Step: {step_id}\n")
                    
                    # 为当前step的所有chunk写入数据
                    for chunk_idx in range(horizon):
                        # 拼接当前chunk的所有action
                        chunk_parts = []
                        for key in available_keys:
                            chunk_data = unnormalized_action[key][batch_idx, chunk_idx, :]
                            chunk_parts.append(chunk_data)
                        
                        # 拼接成一行
                        concatenated = np.concatenate(chunk_parts, axis=0)
                        
                        # 写入数据（空格分隔，每个chunk一行）
                        np.savetxt(f, concatenated.reshape(1, -1), fmt='%.6f', delimiter=' ', comments='')