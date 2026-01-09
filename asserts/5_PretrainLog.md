```yaml
- 0827
 - /vla/users/zhaolin/outputs/Qwen_ft_on_libero_tune_visual_use_pretrainvla_5w/train.log
  - exp-1 使用预训练,训练ViT: bash ft_6_qwen_on_libero_use_pretrain.sh >> ../outputs/Qwen_ft_on_libero_tune_visual_use_pretrainvla_5w/train.log 2>&1

- 0828
  - python scripts/groot_eval_libero.py --server --model_path ../outputs/Qwen_ft_on_libero_tune_visual_use_pretrainvla_5w --embodiment-tag new_embodiment --data-config libero_image_wrist --denoising-steps 4 --backbone-type qwen2_5_vl
    - 使用预训练, 训练ViT的评测结果: INFO:root:Total success rate: 0.86 | checkpoint-50000
    问题: 预训练和微调使用的embodiment id不一致,我们希望去掉embodiment id,转为使用普通的state encoder和action encoder.
  
  - exp-2 使用dino features,训练ViT bash ft_5_qwen_dino_libero.sh | bs16, 25000步等价于bs8的50000步
    ckpt: ../outputs/Qwen_ft_on_libero_tune_visual_random_dit_dino_5w_25000
    - INFO:root:Total success rate: 0.826 | dino embeddings 直接拼接有问题
      思考: 这里训练使用的是 tune_visual, 这样是否有问题? 因为vit训练了, 然后visual tokenizer也训练了, DiT也改动, 这样容易导致分布偏移?
      - 如果引入dino features和visual tokenizer,那么应该不需要tune vit了.
    - 优化方向: dino features在DiT的cross-attention Layer中单独融合，也就是 2 cross-attention layer + 1 self-attn
      或者是把VLM features与dino features一起先通过Perceive融合了
    - 思考: π0未曾使用DiT,也完成了Flow Matching的学习,可以借鉴. lerobot-π0中是使用的self-attention来融合visual-language和actions, 分而治之, 首先self-attention单独学习vl features和
      action features, 然后拼接好后再继续self-attention. 每层反复这样操作.

  - exp-3 使用预训练,不训练ViT的: bash ft_6_qwen_on_libero_use_pretrain.sh >> ../outputs/Qwen_ft_on_libero_no_tune_visual_use_pretrainvla_5w/trainlog.log 2>&1 & 
    - 评测结果: ../outputs/Qwen_ft_on_libero_no_tune_visual_use_pretrainvla_5w_step50000 | Total success rate: 0.78 | 训练50000步,Loss仅下降到0.1左右.
      预训练阶段和微调阶段: VLM都freeze了.

  - exp-4 使用dino features,不训练ViT: bash ft_5_qwen_dino_libero.sh >> ../outputs/Qwen_ft_on_libero_no_tune_visual_random_dit_dino_5w_bs8/trainlog.log 2>&1 &
```


# 2025-09-24,25,26
服务器评测指令: 
(1) `bash eval_6_qwenvla_on_libero.sh 8000 0 ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_object/checkpoint-30000`
(2) `bash start_object.sh 8000 0`, 注意此时`--args.seed 0` (模型评估和Libero-env都设置了 `set_seed_everywhere(seed)` ) 
`python libero_env.py --args.task-suite-name libero_object --args.port $PORT --args.seed 0`

## 评测 libero_object 性能:
`python libero_env.py --args.task-suite-name libero_object --args.port $PORT --args.seed 1`
`bash eval_6_qwenvla_on_libero.sh 8000 0 ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_object/checkpoint-30000`
seed=0 checkpoint-30000 0.948 | checkpoint-35000 0.974 | **checkpoint-40000 0.992** | checkpoint-45000 0.984 | checkpoint-50000 0.982
seed=1 checkpoint-30000 0.954 | checkpoint-35000 0.984 | **checkpoint-40000 0.99** | checkpoint-45000 0.982 | checkpoint-50000 0.984
seed=2 checkpoint-30000 0.954 | checkpoint-35000 0.978 | checkpoint-40000 0.984 | checkpoint-45000 0.988 | checkpoint-50000 0.984
seed=3 checkpoint-30000 0.954 | checkpoint-35000 0.974 | checkpoint-40000 0.99 | checkpoint-45000 0.984 | checkpoint-50000 0.986

## 评测 libero_goal 性能
服务器评测指令: `bash eval_6_qwenvla_on_libero.sh 8000 0 ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_goal/checkpoint-30000`
seed=0 checkpoint-30000 0.95 | checkpoint-35000 0.952 | checkpoint-40000 0.954 | **checkpoint-45000 0.958** | checkpoint-50000 0.954
seed=1 checkpoint-30000 0.946 | checkpoint-35000 0.944 | checkpoint-40000 0.968 | **checkpoint-45000 0.964** | checkpoint-50000 0.962
seed=2 checkpoint-30000 0.942 | checkpoint-35000 0.968 | checkpoint-40000 0.972 | checkpoint-45000 0.96 | checkpoint-50000 0.966
avg:   checkpoint-30000  | checkpoint-35000  | checkpoint-40000 0.965 | checkpoint-45000 0.961 | checkpoint-50000 0.961

本地测试: `python libero_env.py --args.task-suite-name libero_goal --args.port 8000 --args.seed 0`
性能: seed=0和seed=1都是 0.96, seed=2: 0.958

## 评测 libero_spatial 性能
`bash eval_6_qwenvla_on_libero.sh 8000 0 ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_spatial/checkpoint-30000`
env: `python libero_env.py --args.task-suite-name libero_spatial --args.port $PORT --args.seed 0`
第1次: seed=0 checkpoint-30000 0.918 | **checkpoint-35000 0.94** | checkpoint-40000 0.928 | checkpoint-45000 0.932 | checkpoint-50000 0.934 第2次: seed=0 结果同上.
第3次: seed=1 checkpoint-30000 0.922 | checkpoint-35000 0.91 | checkpoint-40000 0.94 | checkpoint-45000 0.936 | checkpoint-50000 0.936
第4次: seed=2 checkpoint-30000 0.94 | checkpoint-35000 0.94 | checkpoint-40000 0.932 | checkpoint-45000 0.926 | checkpoint-50000 0.924
avg:                          0.927                    0.93                    0.933                    0.931                    0.931

## 评测 libero_10 性能
`bash eval_6_qwenvla_on_libero.sh 8004 4 ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w/checkpoint-50000`
env: `python libero_env.py --args.task-suite-name libero_10 --args.port 8006 --args.seed 0`
seed=0 checkpoint-30000 0.888 | checkpoint-35000 0.932 | checkpoint-40000 0.906 | checkpoint-45000 0.914 | checkpoint-50000 0.918
seed=1 checkpoint-30000 0.916 | checkpoint-35000 0.9 | checkpoint-40000 0.914 | checkpoint-45000 0.922 | checkpoint-50000 0.922
seed=2 checkpoint-30000 0.908 | checkpoint-35000 0.932 | checkpoint-40000 0.906 | checkpoint-45000 0.92 | checkpoint-50000 0.91
seed=3 checkpoint-30000 0.872 | checkpoint-35000 0.92 | checkpoint-40000 0.92 | checkpoint-45000 0.922 | checkpoint-50000 0.932

```shell 
2025-09-23
训练libero-机器: 234, 239, 22 
# 116.196.106.22
# 116.198.45.239
# 116.198.44.234

# --model_path ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step50000/raw --embodiment-tag new_embodiment
# 模型性能: 0.932 --args.seed 1 
# --model_path /home/robot/pi/outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_125000_tag_5w_step50000 --embodiment-tag new_embodiment
# 模型性能: 0.862 --args.seed 1 
# --model_path ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_190000_tag_5w_step50000 --embodiment-tag new_embodiment
# 模型性能: 0.91 --args.seed 1 ， 0.922 --args.seed 0 ，0.918 --args.seed 2, 0.928 --args.seed 3, 0.924 seed 0 第2次测试还是 0.924 第3次测试0.91

2025-09-08 
# --model_path ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step45000 --embodiment-tag new_embodiment
# 模型性能: 0.904 0.902 0.906
# --model_path ../outputs/timeactionhead_tune_vit_libero_qwenvla_usepretrain_oxeall_tag_5w_step50000 --embodiment-tag new_embodiment
# 模型性能: 0.916 0.918 0.91 # 模型训练参数: --num-gpus 7 --batch-size 10
# 推理的chunk_size=8: 0.91

2025-09-04: @note: Latent Alignment代码被修改了,还未搭建完成
2025-09-04: 预训练效果差,先解决预训练的问题

2025-09-04 评测使用预训练参数的VLA模型
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-50000 --embodiment-tag new_embodiment
# 性能: 0.84 0.852 0.856
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-45000 --embodiment-tag new_embodiment
# 性能: 0.852 0.848 0.838
# ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w/checkpoint-40000
# 性能: 0.836 0.842 0.85

2025-09-03 评测使用预训练参数的VLA模型:
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_5w_step45000 --embodiment-tag oxe_unified
# 性能: 0.404 | 结果很差,分析:是否是使用相同embodiment_tag的原因?还是因为过拟合了呢. 看训练Loss下降的更低.
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_5w_step30000 --embodiment-tag oxe_unified
# 性能: 0.398 | 对比使用不同的embodiment_tag
# ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w_step15000 --embodiment-tag new_embodiment
# 性能: 0.716
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_5w_20000 --embodiment-tag new_embodiment
# 性能: 0.848
# --model_path ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_usepretrain_tag_5w_step20000 --embodiment-tag new_embodiment
# 性能: 0.76

2025-09-02 eval_4_qwenvla_on_libero.sh 评测模型性能:

# 本地测试, 固定随机种子: set_seed_everywhere(args.seed=7)
# 模型: ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_12layers_5w_step45000 0.892 | 0.896 | 0.912 => 0.9
# 将libero_10的max_steps从520修改为550(因为观察libero的failure video中似乎是操作步长不够?所以测试下改变推理策略),参考UniVLA. 模型不变,性能结果: 0.914 | 0.882(第二次测试0.906) | 0.906 => 0.9006 ？ 0。9086

# 服务器测试: ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_12layers_5w/checkpoint-50000
# 性能: 0.886 | 0.902 | 0.908 => 0.8986
# 服务器测试: ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_12layers_5w/checkpoint-45000
# 性能: 0.912 | 0.9 | 0.888 => 0.9
# 服务器测试: ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_12layers_5w/checkpoint-40000
# 性能: 0.896 | 0.906 | 0.892 => 0.898

# 本地测试 ../outputs/Qwen_ft_on_libero_tune_visual_timeactionhead_5w_step45000
# 0.898 | 0.892 | 0.902 固定seed=7后,性能变化?
# 注释 set_seed_everywhere(args.seed): 0.892 | 0.878 | 0.872

# 改变解噪步长 --denoising-steps 8
# Qwen_ft_on_libero_tune_visual_timeactionhead_5w_step40000 0.888 0.892
# 本地测试,解噪步数=4: 0.886 0.906, 服务器测试:0.904
# libero_env中的seed=7固定(/home/robot/pi/env/libero_sim/libero_env.py) 第1次测试 0.896 | 0.898
# libero_env中的seed=7固定(/home/robot/pi/env/libero_sim/libero_env.py) 第2次测试 0.89 | 0.902

# 新的时间感知动作头: --denoising-steps 4
# Qwen_ft_on_libero_tune_visual_timeactionhead_5w/checkpoint-35000 0.878 0.882 0.882
# Qwen_ft_on_libero_tune_visual_timeactionhead_5w/checkpoint-40000 0.91 0.9 0.89 => 0.9
# Qwen_ft_on_libero_tune_visual_timeactionhead_5w/checkpoint-45000 0.902 0.906 0.888 => 0.898
# Qwen_ft_on_libero_tune_visual_timeactionhead_5w/checkpoint-50000 0.908 0.88 0.882 => 0.89

# Baseline
# qwen-random_action_head-freeze_llm-tune_visual-libero_10_no_noops-max_steps_50000-bs_16 | 0.9

# Dino特征通道融合-性能较差:
# Qwen_ft_on_libero_no_tune_visual_random_dit_dino_5w_bs8_step50000 | 0.672
# 基础Baseline: 
# Qwen_ft_on_libero_tune_visual_random_dit_5w_step3w | 0.862
# Qwen_ft_on_libero_tune_visual_random_dit_5w_step5w | 0.9
```