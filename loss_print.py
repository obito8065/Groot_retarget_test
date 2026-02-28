import json
import matplotlib.pyplot as plt

def load_data_from_json(filepath):
    """
    从指定的JSON文件中加载训练日志数据。
    
    Args:
        filepath (str): JSON文件的路径。
        
    Returns:
        tuple: 包含两个列表的元组 (steps, losses)，
               如果文件不存在或格式错误则返回 ([], [])。
    """
    steps = []
    losses = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 遍历log_history中的每一条记录
            for entry in data.get("log_history", []):
                # 确保记录中同时包含 'step' 和 'loss'
                if 'step' in entry and 'loss' in entry:
                    steps.append(entry['step'])
                    losses.append(entry['loss'])
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{filepath}'")
    except json.JSONDecodeError:
        print(f"错误: 解析JSON文件失败 '{filepath}'")
    except Exception as e:
        print(f"读取文件时发生未知错误 '{filepath}': {e}")
    
    print(f"读取到的Steps数量: {len(steps)}")
    return steps, losses

def plot_loss_comparison(pretrain_data, nopretrain_data):
    """
    绘制pretrain和nopretrain的损失曲线图。
    
    Args:
        pretrain_data (tuple): 包含pretrain的steps和losses列表的元组。
        nopretrain_data (tuple): 包含nopretrain的steps和losses列表的元组。
    """
    pre_steps, pre_losses = pretrain_data
    no_pre_steps, no_pre_losses = nopretrain_data
    
    plt.figure(figsize=(12, 6))
    
    # 绘制nopretrain的loss曲线（蓝色）
    if no_pre_steps and no_pre_losses:
        plt.plot(no_pre_steps, no_pre_losses, color='blue', label='No Pretrain Loss')
    
    # 绘制pretrain的loss曲线（红色）
    if pre_steps and pre_losses:
        plt.plot(pre_steps, pre_losses, color='red', label='Pretrain Loss')
    
    # 设置图表标题和坐标轴标签
    plt.title('Training Loss Comparison: Pretrain vs. No Pretrain')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    
    # 显示图例
    plt.legend()
    
    # 显示网格
    plt.grid(True)
    
    # 保存图表到本地
    plt.savefig('/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_pretrain/n1.5_pretrain_egohuman_fourier_v0.0.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

if __name__ == '__main__':
    # 指定包含完整训练历史的JSON文件路径
    nopretrain_file = '/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_robocasa-perceiver/mutitask1/n1.5_nopretrain_finetune_on_robocasa_tunevl_v1.0/trainer_state.json'
    # pretrain_file = '/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_pretrain/n1.5_pretrain_egohuman_fourier_v0.0/trainer_state.json'
    pretrain_file = "/vla/users/lijiayi/code/GR00T_QwenVLA/outputs_robocasa/mutitask1/n1.5_fourier_pretrain_finetune_on_robocasa_tunevl_fewshot_v3.0/trainer_state.json"
    
    # 从真实的JSON文件中加载数据
    if pretrain_file is not None:
        pretrain_data = load_data_from_json(pretrain_file)  
    if nopretrain_file is not None:
        nopretrain_data = load_data_from_json(nopretrain_file)
    # 绘制图表
    if (pretrain_data[0] and pretrain_data[1]) or \
       (nopretrain_data[0] and nopretrain_data[1]):
        plot_loss_comparison(pretrain_data, nopretrain_data)
    else:
        print("未能从文件中加载任何有效数据，无法生成图表。")