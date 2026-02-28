

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