#!/bin/bash
source /mnt/workspace/envs/conda3/bin/activate robot

python -m tensorboard.main --logdir=$1