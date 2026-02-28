
# 模型微调与性能评估

## 仿真评测

> 测试报告: [Joyspace-Libero测试](https://joyspace.jd.com/pages/KiHqixMWYBjIkQJgnD2G)

> **Libero评测脚本**: `(JD服务器) /vla/users/zhaolin/env/libero_sim/README.md`

- **Libero-Benchmark**-模型评测
```bash
###############################################
1. 终端1 打开模拟器
source /vla/users/envs/conda3/bin/activate libero
cd /vla/users/zhaolin/env/libero_sim/
python libero_env.py --args.task-suite-name libero_10

2. 终端2 运行模型策略
# 注意,请单独拷贝自己的groot环境
# (groot) root@20250213-instance:/vla/users/zhaolin/Isaac-GR00T# pip list | grep gr00t
# gr00t                        1.1.0              /vla/users/zhaolin/Isaac-GR00T
# 服务器的gr00t环境已经与 zhaolin/Isaac-GR00T 绑定.
# 如需使用,请单独安装

/vla/users/zhaolin/Isaac-GR00T# 
conda activate groot 
source /vla/miniconda3/bin/activate groot
export CUDA_VISIBLE_DEVICES=1
python scripts/groot_eval_libero.py \
--server --model_path /vla/users/zhaolin/outputs/ft_flow_libero_long_llm/ft_flow_libero_long_llm_2.5w \
--embodiment-tag new_embodiment --data-config libero_spatial --denoising-steps 4

# 2.5w步 测试结果:(使用了Ego+Wrist图像)
INFO:root:Total success rate: 0.91
INFO:root:Total episodes: 500

###############################################
# **测试单图的效果** | 测试脚本: 
python scripts/groot_eval_libero.py \
--server --model_path /vla/users/zhaolin/outputs/ft_flow_libero_long_llm_single/checkpoint-25000 \
--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4

INFO:root:Total success rate: 0.784
INFO:root:Total episodes: 500
###############################################
# **测试加入Latent的效果**
python scripts/groot_eval_libero.py \
--server --model_path /vla/users/zhaolin/outputs/ft_flow_libero_long_llm_single_with_latent/checkpoint-35000 \
--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4

测试3.5w的结果
INFO:root:Total success rate: 0.808
INFO:root:Total episodes: 500

测试2.5w步的结果:
INFO:root:Total success rate: 0.822
INFO:root:Total episodes: 500
###############################################
# **环境安装**
source /vla/users/envs/conda3/bin/activate libero
conda create -n libero python=3.8.20
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

cd openpi-client
pip install -e .

cd libero
pip install -r requirements.txt
pip install -e .
###############################################
```

- **Libero Benchmark**-模型微调
```bash
(groot) root@20250213-instance:/vla/users/zhaolin/Isaac-GR00T# bash ft_libero_with_latent.sh >> ../outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token/outs_0731.log 2>&1 &
```


## 睿尔曼真机评测

- **睿尔曼机器人数据**-微调版本
```bash  
bash finetune_rm.sh 
```

- **睿尔曼机器人数据**-评测微调模型
```bash 
bash eval_rm_dataset.sh 
```