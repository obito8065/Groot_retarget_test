import torch
import torch.nn as nn

def format_params(num_params: int) -> str:
    """
    将参数数量转换为更易读的格式（B/M/K）
    Args:
        num_params: 参数数量
    Returns:
        格式化后的字符串
    """
    if num_params >= 1e9:  # 十亿
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:  # 百万
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:  # 千
        return f"{num_params / 1e3:.2f}K"
    else:
        return f"{num_params}"


def info_model(model, logger=None):
    # 计算参数量统计
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    # 记录一级模块的训练状态
    trainable_modules = []
    frozen_modules = []
    # 统计所有参数
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
    # 分析一级模块的训练状态
    for name, module in model.named_children():
        is_trainable = any(p.requires_grad for p in module.parameters())
        if is_trainable:
            trainable_modules.append(name)
        else:
            frozen_modules.append(name)
    print("=" * 50)
    print("Model Parameters Statistics:")
    print(f"Total Parameters: {format_params(total_params)}")
    print(
        f"Trainable Parameters: {format_params(trainable_params)} ({trainable_params / total_params * 100:.2f}%)")
    print(f"Frozen Parameters: {format_params(frozen_params)} ({frozen_params / total_params * 100:.2f}%)")
    # # 记录模块训练状态
    print("\nModel Modules Training Status:")
    print("Trainable Modules:")
    for module in trainable_modules:
        print(f"  - {module}")
    print("Frozen Modules:")
    for module in frozen_modules:
        print(f"  - {module}")

    def count_params(module):
        return sum(p.numel() for p in module.parameters())

    def is_trainable(module):
        return any(p.requires_grad for p in module.parameters())

    def format_param_count(count):
        if count >= 1e9:
            return f"{count / 1e9:.3f} B"
        else:
            return f"{count / 1e6:.3f} M"

    print(f"=============== model params ===============")

    for name1, module1 in model.named_children():
        param_count1 = count_params(module1)
        trainable1 = is_trainable(module1)
        print(f"{name1}: {format_param_count(param_count1)}, 可训练: {trainable1}")
        for name2, module2 in module1.named_children():
            param_count2 = count_params(module2)
            trainable2 = is_trainable(module2)
            print(f"  └─ {name1}.{name2}: {format_param_count(param_count2)}, 可训练: {trainable2}")
            if isinstance(module2, nn.ModuleList):
                for name3, module3 in module2.named_children():
                    param_count3 = count_params(module3)
                    trainable3 = is_trainable(module3)
                    print(f"      └─ {name1}.{name2}.{name3}: {format_param_count(param_count3)}, 可训练: {trainable3}")
                    print(f"      └─ {name1}.{name2}.1-{len(module2)}: {format_param_count(param_count3)}, 可训练: {trainable3}")
                    break
            else:
                for name3, module3 in module2.named_children():
                    param_count3 = count_params(module3)
                    trainable3 = is_trainable(module3)
                    print(f"      └─ {name1}.{name2}.{name3}: {format_param_count(param_count3)}, 可训练: {trainable3}")

            if name2 == "0":
                print(f"  └─ {name1}.1-{len(module1)}: {format_param_count(param_count2)}, 可训练: {trainable2}")
                break
    print(f"=============== model params ===============")

    print("=" * 50)

