import torch
from torch.distributions import Beta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set(style="whitegrid")

if __name__ == '__main__':
    # 定义Beta分布参数
    alpha, beta = 1.5, 1.0
    beta_dist = Beta(alpha, beta)

    # 定义其他参数
    batch_size = 100000  # 使用大样本以获得更平滑的分布曲线
    noise_s = 0.999

    # 生成样本
    sample = beta_dist.sample([batch_size])
    result = (noise_s - sample) / noise_s

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 绘制原始Beta分布的样本直方图
    n_bins = 100
    ax1.hist(sample.numpy(), bins=n_bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')

    # 绘制理论Beta分布曲线
    x = np.linspace(0.001, 0.999, 1000)
    beta_pdf = beta_dist.log_prob(torch.tensor(x)).exp().numpy()
    ax1.plot(x, beta_pdf, 'r-', lw=2, label=f'Beta({alpha}, {beta}) PDF')

    ax1.set_title('原始 Beta 分布样本')
    ax1.set_xlabel('值')
    ax1.set_ylabel('概率密度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 绘制变换后的分布
    ax2.hist(result.numpy(), bins=n_bins, density=True, alpha=0.7, color='lightgreen', edgecolor='black')

    # 计算变换后的理论分布
    # 由于 result = (noise_s - sample)/noise_s = 1 - sample/noise_s
    # 我们可以通过变量变换得到理论PDF
    # 设 y = (noise_s - x)/noise_s, 则 x = noise_s*(1-y)
    # f_y(y) = f_x(noise_s*(1-y)) * |dx/dy| = f_x(noise_s*(1-y)) * noise_s
    transformed_pdf = beta_dist.log_prob(torch.tensor(noise_s * (1 - x))).exp().numpy() * noise_s
    ax2.plot(x, transformed_pdf, 'r-', lw=2, label='变换后的理论PDF')

    ax2.set_title('变换后的分布: (noise_s - sample)/noise_s')
    ax2.set_xlabel('值')
    ax2.set_ylabel('概率密度')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 打印一些统计信息
    print(f"原始样本均值: {sample.mean():.4f}, 标准差: {sample.std():.4f}")
    print(f"变换后样本均值: {result.mean():.4f}, 标准差: {result.std():.4f}")