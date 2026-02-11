# 脚本功能：
# 1、将预测的关键点重投影到视频上进行可视化验证
# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证


cd /vla/users/lijiayi/code/groot_retarget

# 1、将预测的关键点重投影到视频上进行可视化验证
python eval_predict_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_taskL_retarget_v5_bs384_horizon100/33ksteps-modify4/100H-33-4.mp4 \
    --keypoints /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260211_161223.txt \
    --fps 30 \
    --steps-per-chunk 100 \
    --radius 8

# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证
python eval_after_retarget_reprojector_cli.py \
    --video   /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_taskL_retarget_v5_bs384_horizon100/33ksteps-modify4/100H-33-4_reprojected.mp4 \
    --retarget /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260211_161225.txt \
    --fps 30 \
    --steps-per-chunk 100 \
    --axis-length 0.05


# 3、对比 predicted keypoints 和 retargeted actions 中的手腕轨迹
python eval_check_pre_retarget_traj.py \
    --pred-file /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50_taskR/24ksteps-modify0/predicted_keypoints_20260210_113800.txt \
    --retarget-file /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50_taskR/24ksteps-modify0/retargeted_actions_20260210_113801.txt \
    --output /vla/users/lijiayi/code/groot_retarget/output_video_record/wrist_trajectory_pred_vs_retarget.png


# 4、可视化左右臂各7个关节的角度变化趋势，并检测IK大角度变化
python eval_visualize_arm_joints.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50_taskR/24ksteps-modify0/robocasa_action_20260210_113801.txt

# 4.1、可视化手部关节轨迹，并检测IK大角度变化（特别关注chunk边界）
python eval_visual_robocasa_hand_joint_traj.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_task2_retarget_v5_bs384_horizon50_taskR/24ksteps-modify0/robocasa_action_20260210_113801.txt \
    --check_ik_jumps \
    --ik_threshold 2.0

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


# 7. 可视化RoboCasa手部关节轨迹

python eval_visual_robocasa_hand_joint_traj.py \
    --action_file /vla/users/lijiayi/code/groot_retarget/output_video_record/robocasa_action_20260206_142705.txt