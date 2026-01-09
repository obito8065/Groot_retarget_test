import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')  # 切换到文件系统共享，基于临时文件而非FD
# 规避 [rank7]: OSError: [Errno 24] Too many open files: '/tmp/tmp7olgnlrk/__pycache__' 错误
import os
import subprocess
import sys
import warnings
warnings.filterwarnings("ignore")
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import torch
from torch.utils.data import random_split
import tyro
from transformers import TrainingArguments

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset, EpochAwareSubset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP, UnifiedRobotDataConfig, DATASET_WEIGHTS
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model
from gr00t.argsconfig import ArgsConfig


#####################################################################################
# main training function
#####################################################################################

def load_finetune_dataset(config: ArgsConfig):
    ## GR00T N-1.5 use the same transforms, modality configs, and embodiment tag for all datasets here,
    ## in reality, you can use dataset from different modalities and embodiment tags
    ## Ours: use different transforms
    ## 1.1 modality configs and transforms
    ## 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:

        if isinstance(config.embodiment_tag, list):
            embodiment_tag = EmbodimentTag(config.embodiment_tag[0])
        elif isinstance(config.embodiment_tag, str):
            embodiment_tag = EmbodimentTag(config.embodiment_tag)
        else:
            raise NotImplementedError

        # TODO: ActionHead使用相同的Embodiment_Tag,微调性能变差
        # if config.load_pretrained:
        #     embodiment_tag = EmbodimentTag("oxe_unified")

        data_config_cls = DATA_CONFIG_MAP[config.data_config[0]] # default： egohuman dataset config
        modality_configs = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
            check_stats=False,
            instr_use_episode_index=config.instr_use_episode_index,
        )
    else:
        single_datasets = []
        if not (len(config.dataset_path) == len(config.data_config) == len(config.embodiment_tag)):
            raise ValueError(f"Mismatched lengths: dataset_path({len(config.dataset_path)}), "
                             f"data_config({len(config.data_config)}), "
                             f"embodiment_tag({len(config.embodiment_tag)})")
        
        for p, q, r in zip(config.dataset_path, config.data_config, config.embodiment_tag):
            assert os.path.exists(p), f"Dataset path {p} does not exist"

            embodiment_tag = EmbodimentTag(r)
            data_config_cls = DATA_CONFIG_MAP[q]
            modality_configs = data_config_cls.modality_config()
            transforms = data_config_cls.transform()
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                check_stats=False,
                instr_use_episode_index=config.instr_use_episode_index,
            )
            single_datasets.append(dataset)

        if config.balance_dataset_weights:
            # 为True时：按数据集大小比例混合（原来的默认行为）
            train_dataset = LeRobotMixtureDataset(
                data_mixture=[
                    (dataset, 1.0)
                    for dataset in single_datasets
                ],
                mode="train",
                balance_dataset_weights=True,
                balance_trajectory_weights=config.balance_trajectory_weights,
                seed=42,
                metadata_config={
                    "percentile_mixing_method": "weighted_average",
                },
            )
            print(f"Loaded {len(single_datasets)} datasets with balance_dataset_weights=True")
        else:
            # 为False时：按照robot：egodex为1:1混合
            # 直接使用 config.embodiment_tag，不需要保存额外的列表
            egodex_weight = 1.0
            robot_weight = 4000.0 / 150.0  # 约 26.67
            
            data_mixture = []
            for idx, (dataset, emb_tag) in enumerate(zip(single_datasets, config.embodiment_tag)):
                # 直接使用 config.embodiment_tag 中的原始字符串
                if emb_tag == "new_embodiment":
                    weight = egodex_weight
                else:  # robocasa 或其他机器人数据
                    weight = robot_weight
                data_mixture.append((dataset, weight))
                print(f"Dataset {idx}: {emb_tag}, weight={weight:.4f}")

            train_dataset = LeRobotMixtureDataset(
                data_mixture=data_mixture,
                mode="train",
                balance_dataset_weights=False,
                balance_trajectory_weights=config.balance_trajectory_weights,
                seed=42,
                metadata_config={
                    "percentile_mixing_method": "weighted_average",
                },
            )
            print(f"Loaded {len(single_datasets)} datasets with balance_dataset_weights=False (robot:egodex=1:1)")

    return train_dataset
  

def load_pretrain_dataset(config: ArgsConfig):
    embodiment_tag = EmbodimentTag("oxe_unified")
    if len(config.dataset_path) == 1:
        dataset_type = Path(config.dataset_path[0]).stem
        data_config_cls = UnifiedRobotDataConfig(dataset_type=dataset_type)
        modality_configs = data_config_cls.modality_config()
        transforms = data_config_cls.transform()
        train_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
            check_stats=False,
            instr_use_episode_index=False,
        )
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            dataset_type = Path(p).stem

            data_config_cls = UnifiedRobotDataConfig(dataset_type=dataset_type)
            modality_configs = data_config_cls.modality_config()
            transforms = data_config_cls.transform()
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
                check_stats=False,
                instr_use_episode_index=False,
            )
            single_datasets.append(dataset)

        data_mixture = []
        for sub_dataset in single_datasets:
            dataset_weight = DATASET_WEIGHTS[sub_dataset.dataset_name]
            data_mixture.append(
                (sub_dataset, dataset_weight)
            )

        train_dataset = LeRobotMixtureDataset(
            data_mixture=data_mixture,
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
            merge_metadata=False,
            pretrain=True,
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")

    return train_dataset

def main(config: ArgsConfig):
    """Main training function."""
    # ------------ step 1: load dataset ------------
    if not config.pretrain:
        train_dataset = load_finetune_dataset(config)
    else:
        train_dataset = load_pretrain_dataset(config)

    eval_strategy = "no"
    eval_steps = None
    eval_dataset = None
    if config.use_eval:
        total_len = len(train_dataset)
        eval_ratio = 0.1
        eval_len = int(total_len * eval_ratio)
        train_len = total_len - eval_len
        g = torch.Generator().manual_seed(42)
        train_subset, eval_subset = random_split(train_dataset, [train_len, eval_len], generator=g)
        train_dataset = EpochAwareSubset(train_subset.dataset, train_subset.indices)
        eval_dataset = EpochAwareSubset(eval_subset.dataset, eval_subset.indices)
        eval_strategy = "steps"
        eval_steps = config.save_steps
        # eval_steps = 10 # for debug


    if config.use_time_aware_action_head:
        from gr00t.model.action_head.time_aware_action_head import TimeAwareActionHead_Config
        action_head_config = TimeAwareActionHead_Config()
        time_aware_action_head_config = action_head_config.convert_to_dict()
        time_aware_action_head_config['action_dim'] = config.action_dim
        time_aware_action_head_config['layers'] = config.action_layers
        time_aware_action_head_config['heads'] = config.action_heads
    else:
        time_aware_action_head_config = None

    # ------------ step 2: load model ------------
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
        backbone_type=config.backbone_type,  # backbone type, currently only support qwen2_5_vl, eagle2
        backbone_model_name_or_path=config.backbone_model_name_or_path,
        use_dino=config.use_dino,
        use_time_aware_action_head=config.use_time_aware_action_head,
        time_aware_action_head_config=time_aware_action_head_config,
        load_pretrained=config.load_pretrained,
    )

    if config.update_backbone and config.backbone_type and config.backbone_model_name_or_path:
        print(f"================== VLA Update Backbone ====================")
        model.update_backbone(
            backbone_type=config.backbone_type,
            backbone_model_name_or_path=config.backbone_model_name_or_path,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,  # select_layer=1 means we will use the first layer of the backbone
        )

    if config.update_action_head and not config.use_time_aware_action_head:
        print(f"================== VLA Update Action Head ====================")
        model.update_action_head(model.config, dit_num_layers=config.dit_num_layers, action_dim=config.action_dim)



    # --- Save full architecture to OUTPUT_DIR (always, rank0 only) ---
    if os.environ.get("RANK", "0") == "0":
        def walk_module(m, prefix=""):
            lines = []
            for name, child in m.named_children():
                lines.append(f"{prefix}{name}: {child.__class__.__name__}")
                lines.extend(walk_module(child, prefix + "  "))
            return lines

        arch_repr = repr(model)  # PyTorch 模块层级字符串
        tree_str = "\n".join(walk_module(model))
        dump = f"=== MODEL REPR ===\n{arch_repr}\n\n=== MODULE TREE ===\n{tree_str}\n"

        os.makedirs(config.output_dir, exist_ok=True)
        out_file = os.path.join(config.output_dir, "model_arch.txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(dump)
        print(f"Saved model architecture to: {out_file}")


     
    if config.enable_latent_embedding:
        model.enable_latent_alignment(
            M=256, hidden_dim=1536
        )
        train_dataset.update_configs(enable_latent_embedding=config.enable_latent_embedding)

        if config.enable_last_action_tokens:
            model.action_head.update_configs(enable_last_action_tokens=config.enable_last_action_tokens)

        print(f"####### enable latent embedding ########\n"
              f"enable_last_action_tokens={config.enable_last_action_tokens}\n")

    if config.use_dino:
        model.update_dino_vit()

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )

    # 检查模型的训练参数和冻住参数
    from scripts.train_utils import info_model
    info_model(model)

    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=None,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,  # Increase from 1 to 4 to maintain effective batch size
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=config.num_train_epochs, # 300,
        max_steps=config.max_steps, # If > 0: set total number of training steps to perform. Override num_train_epochs.
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy=eval_strategy, # no evaluation during training, use "steps" to evaluate at specified steps
        eval_steps=eval_steps, # if eval, use same as save_steps
        save_total_limit=100,
        report_to=config.report_to,
        seed=42,
        do_eval=True,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
        use_dino=config.use_dino
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # # Remove any existing CUDA_VISIBLE_DEVICES from environment
            # if "CUDA_VISIBLE_DEVICES" in os.environ:
            #     del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                f"--master_port={config.master_port}",
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
