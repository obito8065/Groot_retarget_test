# GR00t-n1.5在libero上的微调和评估

## 仿真评测

- **Libero-Benchmark**-模型评测
```bash
###############################################
#以 libero-long为例：
1. 终端1 打开模拟器
source /vla/users/envs/conda3/bin/activate libero
cd /vla/users/zhaolin/env/libero_sim/
bash start_10.sh 8000(端口号) 0(GPU序号) 
# 上述start_10.sh脚本会调用 ：python libero_env.py --args.task-suite-name libero_10

2. 终端2 运行模型策略
bash eval_7_grootn15_on_libero.sh 8090 0 ./outputs/libero_object/n1.5_nopretrain_finetune_on_sample_libero_10_v2.1/checkpoint-8000
# 每个GPU可以测试一个模型

```

