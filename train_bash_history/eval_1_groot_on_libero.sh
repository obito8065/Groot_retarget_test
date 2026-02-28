#!/bin/bash

######################################################################
# [本地评测指令]
# (1) 模型推理
# python scripts/groot_eval_libero.py \
# --server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_2.5w \
# --embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 \
# --enable_latent_embedding # 将Latent Alignment加入到模型中,对齐DiT的Representations
# (2) 仿真环境模拟器 (libero)$ python /home/robot/pi/env/openpi/libero_env.py --args.task-suite-name libero_10
######################################################################


# 测试结果: 80.2% 2.5w步
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token_2.5w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 --enable_latent_embedding

# INFO:root:Total success rate: 0.784
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token_2w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 --enable_latent_embedding

# INFO:root:Total success rate: 0.808
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token_3w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 --enable_latent_embedding

# INFO:root:Total success rate: 0.78
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token_3.5w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 --enable_latent_embedding

# INFO:root:Total success rate: 0.772
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_with_latent_use_right_token_4w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4 --enable_latent_embedding

# GR00T Libero 无Representation Alignment, train LLM+Action Head
# INFO:root:Total success rate: 0.806
#python scripts/groot_eval_libero.py \
#--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_3w \
#--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4

# INFO:root:Total success rate: 0.79
python scripts/groot_eval_libero.py \
--server --model_path /home/robot/pi/outputs/ft_flow_libero_long_llm_single_3.5w \
--embodiment-tag new_embodiment --data-config libero_single --denoising-steps 4