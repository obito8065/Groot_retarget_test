# 脚本功能：
# 1、将预测的关键点重投影到视频上进行可视化验证
# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证


cd /vla/users/lijiayi/code/groot_retarget

# 1、将预测的关键点重投影到视频上进行可视化验证
python eval_predict_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep/24ksteps-modify3/3.mp4 \
    --keypoints /vla/users/lijiayi/code/groot_retarget/output_video_record/predicted_keypoints_20260129_171323.txt \
    --fps 30 \
    --steps-per-chunk 16 \
    --radius 8

# 2、将重投影后的视频与retargeted_actions进行对齐，并可视化验证
python eval_after_retarget_reprojector_cli.py \
    --video /vla/users/lijiayi/code/groot_retarget/output_video_record/output_retarget_1tasks_1000ep/n1.5_nopretrain_finetuneALL_on_robocasa_eepose_retarget_v3_300ep/24ksteps-modify3/3_reprojected.mp4 \
    --retarget /vla/users/lijiayi/code/groot_retarget/output_video_record/retargeted_actions_20260129_171324.txt \
    --fps 30 \
    --steps-per-chunk 16 \
    --axis-length 0.05