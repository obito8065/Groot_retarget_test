# 脚本功能：
# 1、将预测的关键点重投影到视频上进行可视化验证
# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证


cd /vla/users/lijiayi/code/groot_retarget

# 1、将预测的关键点重投影到视频上进行可视化验证
python eval_predict_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify7/7.mp4 \
    --keypoints /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260204_172644.txt \
    --fps 30 \
    --steps-per-chunk 50 \
    --radius 8

# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证
python eval_after_retarget_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50_task2/33ksteps-modify7/7_reprojected.mp4 \
    --retarget /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260204_172645.txt \
    --fps 30 \
    --steps-per-chunk 50 \
    --axis-length 0.05


# 3、对比 predicted keypoints 和 retargeted actions 中的手腕轨迹
python eval_check_pre_retarget_traj.py \
    --pred-file /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260203_095724.txt \
    --retarget-file /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260203_095725.txt \
    --output /vla/users/lijiayi/code/groot_retarget/output_video_record/wrist_trajectory_pred_vs_retarget.png


# 4、可视化左右臂各7个关节的角度变化趋势
python eval_visualize_arm_joints.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260130_150142.txt

# 5. 重投影可视化推理后执行的关节FK后的关键点验证

python check_robocasa_hand_joint_keypoint.py \
    --action-txt /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/robocasa_action_20260202_152421.txt \
    --video-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/4_reprojected_eepose_reprojected.mp4 \
    --robot-urdf /vla/users/lijiayi/code/groot_retarget/gr00t/eval/robot_assets/GR1T2/urdf/GR1T2_fourier_hand_6dof.urdf \
    --output-path /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep_horizon50/37ksteps-modify4/output_reprojected2.mp4 \
    --fps 30.0

# 6. 测试纯retarget API效果
python /vla/users/lijiayi/code/groot_retarget/gr00t/eval/fourier_hand_retarget_api_test.py \
    --parquet_file /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v3/data/chunk-000/episode_000000.parquet \
    --action_key observation.state \
    --output_plot /vla/users/lijiayi/code/groot_retarget/output_video_recordretarget_comparison.png \
    --max_frames 1000

python /vla/users/lijiayi/code/groot_retarget/gr00t/eval/fourier_hand_retarget_api_test.py \
    --parquet_file /vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300_keypoints_v3/data/chunk-000/episode_000000.parquet \
    --action_key action \
    --output_plot /vla/users/lijiayi/code/groot_retarget/output_video_recordretarget_comparison.png \
    --max_frames 1000