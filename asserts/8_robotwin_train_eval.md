# GR00t-n1.5在robotwin上的微调和评估

## 仿真评测

### 1.配置notebook上的环境依赖：
    查看 https://joyspace.jd.com/pages/VeX19BDV4iVj33JziqX1，
    此外，使用robotwin镜像也会出现vk::PhysicalDevice::createDeviceUnique: ErrorInitializationFailed，同样需要按照groot镜像中的安装脚本进行安装

### 2.模型训练+评测：
```bash
sft_robotwin_150ep_3task
    |--train_sh
        |-- sft_1_robotwin_nopretrain_finetuneALL_150ep_grootn15_eepose.sh

    |--eval_sh # （多任务评测脚本）
        |-- eval_sh/1_batch_eval_groot_robotwin.sh

```

### 3.评测脚本

#### 3.1 单任务评测：
```bash
# 1. 运行policy server
cd GR00T_QwenVLA
bash scripts/1_singletask_groot_robotwin_server.sh
```
服务端实际会调用：
    1. model_path 训练的ckpt
    2. dataconfig 要与数据集定义的相同
    3. embodimenttag 与数据集定义的相同
    4. port 端口号


```bash
# 2.运行仿真client：
cd /mnt/workspace/users/lijiayi/RoboTwin_utils/RoboTwin/
bash script/1_single_task_groot_client.sh
```
客户端实际会调用：
    1. task_name 评测任务名
    2. task_config 所有easy任务都是demo_clean
    3. seed 随机种子
    4. port 保证和server一致
    5. save_dir 视频保存位置


#### 3.2 多任务评测：
```bash
# 一键启动server：1v1 client的多任务评测脚本：
bash /mnt/workspace/users/lijiayi/GR00T_QwenVLA/scripts/1_batch_eval_groot_robotwin.sh

```

服务端运行脚本：
    1. base port：默认5个client对应1个server进行交互，简化server端冗余，baseport要和客户端一致
    2. model_path：模型路径
    3. data_config，embodiment_tag：与数据集一致
    4. save_dir：服务端输出log
    5. repo_root：GR00T_QwenVLA路径
    6. task_config：easy任务都使用demo_clean
    7. seed_base： 随机种子初始值
    8. gpu_ids： 可用gpu
    9. tasks_per_gpu：每个gpu多少个任务评测
    10. n_episodes： 每个任务评测多少轮
    11. use_eepose： 使用endpose

```bash
cd /mnt/workspace/users/lijiayi/GR00T_QwenVLA

python3 scripts/batch_robotwin_eval.py \
    --model_path /mnt/workspace/users/lijiayi/GR00T_QwenVLA/output_ckpt/output_robotwin_easy_ckpt_3tasks/n1.5_nopretrain_finetuneALL_on_robotwin_eepose_v0.2/checkpoint-6390 \
    --data_config robotwin_ego \
    --embodiment_tag robotwin \
    --task_names adjust_bottle place_burger_fries \
    --task_config demo_clean \
    --base_port 11016 \
    --seed_base 0 \
    --save_dir /mnt/workspace/users/lijiayi/GR00T_QwenVLA/outputs_robotwin/eval_result_3task_eepos_test \
    --gpu_ids 0 1 \
    --tasks_per_gpu 1 \
    --n_episodes 20 \
    --use_eepose

# 此外还需要在/GR00T_QwenVLA/scripts/batch_robotwin_eval.py中修改两个默认的服务端和客户端的路径
parser.add_argument('--policy_root', type=str, default="/mnt/workspace/users/lijiayi/GR00T_QwenVLA")
parser.add_argument('--robotwin_root', type=str, default="/mnt/workspace/users/lijiayi/RoboTwin_utils/RoboTwin")

```
