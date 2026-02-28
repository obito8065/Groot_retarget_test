dataset_list=(
    "/home/robot/pi/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-apple/"
    "/home/robot/pi/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-pear/"
    "/home/robot/pi/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-grapes/"
    "/home/robot/pi/PhysicalAI-Robotics-GR00T-Teleop-G1/g1-pick-starfruit/"
)

python scripts/gr00t_finetune.py \
  --dataset-path ${dataset_list[@]} \
  --num-gpus 1 --batch-size 95  --output-dir ~/checkpoints/full-g1-mix-fruits/  \
  --data-config unitree_g1 --max-steps 15000 --base_model_path ../GR00T-N1.5-3B --report_to tensorboard --video-backend torchvision_av \