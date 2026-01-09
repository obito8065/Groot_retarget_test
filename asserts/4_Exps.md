## 模型性能实验记录



- 0718 | **测试gr00t n 1.5在Libero上的结果**
    - 评测结果: 0.878 (500 episodes, Libero-Long);
    - 模型: Ego+Wrist, 只训练Action Head (`ft_flow_libero_long_1w`)

```bash
# sim
source ~/miniconda3/bin/activate libero
python /home/robot/pi/env/openpi/libero_env.py --args.task-suite-name libero_10

# 4w
python scripts/groot_eval_libero.py \
--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_4w \
--embodiment-tag new_embodiment --data-config libero_spatial --denoising-steps 4
# 4w测试结果
INFO:root:████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉            | 47/50 [03:48<00:13,  4.39s/it]
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 48...
INFO:root:Success: True
INFO:root:# episodes completed so far: 498
INFO:root:# successes: 432 (86.7%)

INFO:root:████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉        | 48/50 [03:52<00:08,  4.39s/it]
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 49...
INFO:root:Success: True
INFO:root:# episodes completed so far: 499
INFO:root:# successes: 433 (86.8%)

INFO:root:████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▉    | 49/50 [03:57<00:04,  4.53s/it]
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 50...
INFO:root:Success: True
INFO:root:# episodes completed so far: 500
INFO:root:# successes: 434 (86.8%)
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:02<00:00,  4.85s/it]
INFO:root:Current task success rate: 0.92██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [04:02<00:00,  4.59s/it]
INFO:root:Current total success rate: 0.868
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [45:07<00:00, 270.73s/it]
INFO:root:Total success rate: 0.868
INFO:root:Total episodes: 500



######### policy 1w
python scripts/groot_eval_libero.py \
--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_1w \
--embodiment-tag new_embodiment --data-config libero_spatial --denoising-steps 4
# 测试结果
 92%|█████████▏| 46/50 [05:29<00:26,  6.60s/it]INFO:root:
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 47...
INFO:root:Success: True
INFO:root:# episodes completed so far: 497
INFO:root:# successes: 436 (87.7%)

 94%|█████████▍| 47/50 [05:35<00:19,  6.45s/it]INFO:root:
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 48...
INFO:root:Success: True
INFO:root:# episodes completed so far: 498
INFO:root:# successes: 437 (87.8%)

 96%|█████████▌| 48/50 [05:41<00:12,  6.24s/it]INFO:root:
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 49...
INFO:root:Success: True
INFO:root:# episodes completed so far: 499
INFO:root:# successes: 438 (87.8%)

 98%|█████████▊| 49/50 [05:47<00:06,  6.30s/it]INFO:root:
Task: put the yellow and white mug in the microwave and close it
INFO:root:Starting episode 50...
INFO:root:Success: True
INFO:root:# episodes completed so far: 500
INFO:root:# successes: 439 (87.8%)

100%|██████████| 50/50 [05:53<00:00,  7.07s/it]
INFO:root:Current task success rate: 0.88
INFO:root:Current total success rate: 0.878
100%|██████████| 10/10 [1:01:34<00:00, 369.47s/it]
INFO:root:Total success rate: 0.878
INFO:root:Total episodes: 500
```

- 0717 **睿尔曼机器人训练**
    - 消融实验,训练解冻ViT、解冻LLM、完全解冻全参数：
```shell 
bash ft_vlm_flow.sh # 脚本中具有解冻网络模块的参数
# - 目前全参数训练,灵巧手的抓取+位置泛化性能较好
# - 冻住VLM,instruction-following的能力较好
```

- 0717 **Libero数据集训练**
```shell 
bash ft_on_libero_long.sh # 训练lerobot libero long的数据集
```

- **评测Libero指令记录**
```bash 
# 运行仿真器
source ~/miniconda3/bin/activate libero

(libero) robot@robot:~/pi/Isaac-GR00T$ python /home/robot/pi/env/openpi/libero_env.py --args.task-suite-name libero_10

# 运行模型开始评测
(groot) robot@robot:~/pi/Isaac-GR00T$ python scripts/groot_eval_libero.py --server --model_path /home/robot/pi/outputs/ft_flow_libero_long_4w --embodiment-tag new_embodiment --data-config libero_spatial --denoising-steps 4
```