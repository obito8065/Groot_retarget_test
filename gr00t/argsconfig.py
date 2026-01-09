from typing import List, Literal
from dataclasses import dataclass
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING

@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: List[Literal[tuple(DATA_CONFIG_MAP.keys())]] = "fourier_gr1_arms_waist"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""

    # Training parameters
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = -1
    """Maximum number of training steps. If > 0: set total number of training steps to perform. Override num_train_epochs."""

    num_train_epochs: int = -1

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 1000
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard"] = "none"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard')."""

    # Data loading parameters
    embodiment_tag: List[Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())]] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

    ########### 自定义参数 ##############
    use_multi_size_img: bool = False

    action_horizon: int = 16
    """Action horizon to evaluate."""

    ddp_find_unused_parameters: bool = False

    # raw gr00t code: instr_use_episode_index=False
    instr_use_episode_index: bool = True # tasks.jsonl use episode index, otherwise task index

    enable_latent_embedding: bool = False
    """Whether to enable latent embedding. (SigLip2)"""

    enable_last_action_tokens: bool = False # 采用末尾tokens预测action

    reinitialize_action_head: bool = False # 重新初始化DiT

    merge_metadata: bool = False # 预训练不合并多个数据集的metadata

    ########### qwen2.5-vl ##############
    backbone_type: str = None
    """VLM backbone. [eagle2, eagle2_5, qwen2_5_vl]"""

    backbone_model_name_or_path: str = None
    """Model name or path for VLM backbone."""

    select_layer: int = -1
    """Layer index for VLM feature."""

    master_port: int = 65432

    update_backbone: bool = False
    """Whether to update the backbone model during training. If True, the backbone model will be updated with the new model name or path."""

    update_action_head: bool = False
    """Whether to update the action head during training. If True, the action head will be initialized randomly."""

    dit_num_layers: int = -1
    """Layer num for DiT. -1 means use the default setting (16)."""

    ########### pretrain #############
    pretrain: bool = False

    use_dino: bool = False

    use_time_aware_action_head: bool = False

    action_dim: int = 1536
    action_heads: int = 16
    action_layers: int = 16

    load_pretrained: bool = False

    use_eval: bool = False
    """Whether evaluate the model during training."""
