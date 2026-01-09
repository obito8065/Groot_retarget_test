# GR00t-n1.5在robocasa上的微调和评估

## 仿真评测

- **Robocasa-Benchmark**-模型评测
### 1. 8台服务器上运行：
```bash
1. 终端1 运行server，运行模型策略，等待仿真器的输出：
cd GR00T_QwenVLA
conda activate gr00t_p
python3 scripts/inference_service.py \
--server --model_path ./outputs_robocasa/task1/n1.5_nopretrain_finetune_on_robocasa_tunevl_v1/checkpoint-16000  --port 8813 --data_config fourier_gr1_arms_waist  --embodiment_tag robocasa
# 使用上次训练好的embodiment_tag的mlp，不同构型切换不同tag-mlp

2. 终端2 打开模拟器
cd GR00T_QwenVLA
conda activate robocasa
python3 scripts/simulation_service.py --client     --env_name gr1_unified/PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_Env    --max_episode_steps 720     --port 8813     --n_envs 5     --n_episodes 30     --video_dir ./outputs_robocasa/videos/groot-task1-nopretrain-16K
# n_envs代表五个仿真器并行推理，n_episodes任务测试次数，video_dir测试仿真视频保存路径
```


### 2. 运行多任务评测
```bash
1. 集群先在自己notebook配置EGL库，老服务器不需要
https://joyspace.jd.com/pages/Uo9FPnQAHMxrg3Y10IqY

2. 运行多进程评测：
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA/scripts
# (1) 测评GR00t-n15的使用默认的joint数据 for robocasa:
bash start_robocasa_multitask.sh 
# (2) # 测评GR00t-n15的对齐动作空间 for robocasa: （动作对齐以来kdl库，参考后面配置kdl库）
bash start_robocasa_multitask_eepose.sh 
# (3) 测评GR00t-n15的resume gr1 Actionhead for robocasa:
bash start_robocasa_multitask_resumeAH.sh 

#每次评测后手动杀掉服务端进程：
bash kill_robocasa_eval.sh
```

## 3. 补充安装依赖
```bash
# 1. 配置kdl库（之前先配置pip包）
bash /GR00T_QwenVLA/set_package/set_kdl.sh
bash /GR00T_QwenVLA/set_package/set_pykdl.sh

```