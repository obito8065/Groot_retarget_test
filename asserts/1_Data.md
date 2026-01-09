# 1. 训练数据说明:
- 数据集格式如下: 
```bash 
datasets/take_cup_lerobot62$ tree -L 2
.
├── data
│ └── chunk-000
├── meta
│ ├── episodes.jsonl
│ ├── info.json
│ ├── modality.json
│ ├── stats.json
│ └── tasks.jsonl
└── videos
    └── chunk-000

----
modality.json 数据如下: (已存放在 infos/demo_meta_rm_take_cup/modality.json 路径下)
{
    "state":{
        "arm_right_position_state":{
            "start":18,
            "end":21
        },
        "arm_right_axangle_state":{
            "start":21,
            "end":24
        },
        "hand_right_pose_state":{
            "start":48,
            "end":54
        }
    },
    "action":{
        "arm_right_position_action":{
            "start":6,
            "end":9
        },
        "arm_right_axangle_action":{
            "start":9,
            "end":12
        },
        "hand_right_pose_action":{
            "start":18,
            "end":24
        }
    },
    "video":{
        "cam_head_left": {
            "original_key": "observation.images.cam_head_left"
            }
    },
    "annotation":{
        "human.action.task_description":{
            "original_key": "task_index"
        }
    }
}
```