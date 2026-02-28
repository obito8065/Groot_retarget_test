# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Tuple
import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tree
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel, GPT2Config
from transformers.feature_extraction_utils import BatchFeature

from .action_head.flow_matching_action_head import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)
from .backbone import EagleBackbone
from .backbone import Qwen2_5_VL_Backbone

from fusevla.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
from fusevla.models.visual_tokenizer import PerceiverResampler

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3


class SimpleMLP(nn.Module):
    """2-layer MLP for embeddings and decoders."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# config
@dataclass
class GR00T_N1_5_Config(PretrainedConfig):
    model_type = "gr00t_n1_5"
    backbone_cfg: dict = field(init=False, metadata={"help": "Backbone configuration."})

    action_head_cfg: dict = field(init=False, metadata={"help": "Action head configuration."})

    action_horizon: int = field(init=False, metadata={"help": "Action horizon."})

    action_dim: int = field(init=False, metadata={"help": "Action dimension."})
    compute_dtype: str = field(default="float32", metadata={"help": "Compute dtype."})

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


# real model
class GR00T_N1_5(PreTrainedModel):
    supports_gradient_checkpointing = True
    config_class = GR00T_N1_5_Config
    """
    we expect the backbone output to have a key 'backbone_features' with shape (batch_size, n, hidden_size)
    here n is variable and can be e.g. time, 1 or user specified
    we expect the action head output to have a key 'action_pred' with shape (batch_size, time, action_dim) during inference time
    we expect these to have type BatchFeature, and they can of course have many other user specified keys too
    """

    def __init__(
        self,
        config: GR00T_N1_5_Config,
        local_model_path: str,
        enable_latent_alignment: bool = False,
        backbone_type: str = "eagle",
        backbone_model_name_or_path: str = None,
        select_layer: int = 12,
        use_dino: bool = False,
        use_time_aware_action_head: bool = False,
    ):
        assert isinstance(config.backbone_cfg, dict)
        assert isinstance(config.action_head_cfg, dict)

        super().__init__(config)
        self.local_model_path = local_model_path

        if use_time_aware_action_head:
            config.action_head_cfg['use_time_aware_action_head'] = use_time_aware_action_head

        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)

        if isinstance(backbone_type, str) and "qwen" in backbone_type.lower():
            if not "model_name_or_path" in config.backbone_cfg:
                config.backbone_cfg["model_name_or_path"] = backbone_model_name_or_path
            self.backbone = Qwen2_5_VL_Backbone(**config.backbone_cfg)
            if "eagle_path" in self.config.backbone_cfg:
                del self.config.backbone_cfg["eagle_path"]
        else:
            self.backbone = EagleBackbone(**config.backbone_cfg)

        self.action_horizon = config.action_horizon
        self.action_dim = config.action_dim
        self.compute_dtype = config.compute_dtype

        if enable_latent_alignment:
            self.enable_latent_alignment(M=256, hidden_dim=1536)
        else:
            self.vl_embedding_model = None
            self.future_tokens = None
            self.embedding_decode = nn.Identity()

        self.use_dino = use_dino
        if use_dino:
            vis_dim = 2048
            vis_latent = 128
            DEFAULT_DINO_PATH = os.environ.get("DEFAULT_DINO_PATH")
            self.dino_path = str(DEFAULT_DINO_PATH)
            dino_vit_config = AutoConfig.from_pretrained(self.dino_path, trust_remote_code=True)
            self.dino_vit = DINOv3ViTModel(dino_vit_config)
            # self.vit_linear = torch.nn.Linear(1024, vis_dim)

            self.vision_tokenizer = PerceiverResampler(
                dim=1024,
                dim_inner=1024,
                num_latents=64,
                depth=6
            )

            self.vit_linear = torch.nn.Linear(1024, vis_dim)


    def update_dino_vit(self):
        del self.dino_vit
        print(f"########### init dino v3 model ###########")
        self.dino_vit = DINOv3ViTModel.from_pretrained(
            self.dino_path, trust_remote_code=True
        )
        # 冻结整个DINOv3ViT模型
        for param in self.dino_vit.parameters():
            param.requires_grad = False
        for param in self.vit_linear.parameters():
            param.requires_grad = True
        for param in self.vision_tokenizer.parameters():
            param.requires_grad = True

    def update_action_horizon(self, action_horizon):
        self.action_horizon = action_horizon
        self.config.action_horizon = action_horizon
        self.config_class.action_horizon = action_horizon
        self.action_head.update_action_horizon(action_horizon)
        self.base_model.config.action_horizon = action_horizon
        self.base_model.config.action_head_cfg['action_horizon'] = action_horizon
        print(f"\n================== Update GR00T_N1_5 action_horizon = {action_horizon} ====================\n")

    def update_backbone(self, backbone_type, backbone_model_name_or_path, tune_llm=False, tune_visual=False, select_layer=12):
        del self.backbone
        project_to_dim = None
        if "qwen" in backbone_type:
            if "7b" in os.environ.get("DEFAULT_EAGLE_PATH", "").lower():
                project_to_dim = 3584
            self.backbone = Qwen2_5_VL_Backbone(
                model_name_or_path=backbone_model_name_or_path,
                tune_llm=tune_llm,
                tune_visual=tune_visual,
                select_layer=select_layer,
                project_to_dim=project_to_dim,
                load_from_pretrained=True,
            )
        elif "eagle" in backbone_type.lower():
            if "2_5" in backbone_type.lower():
                project_to_dim = 3584
            else:
                project_to_dim = 1536
            self.backbone = EagleBackbone(
                model_name_or_path=backbone_model_name_or_path,
                tune_llm=tune_llm,
                tune_visual=tune_visual,
                select_layer=select_layer,
                project_to_dim=project_to_dim,
                load_from_pretrained=True,
            )
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        self.config.backbone_cfg["model_name_or_path"] = backbone_model_name_or_path
        self.config.backbone_cfg["backbone_type"] = backbone_type
        self.config.backbone_cfg["project_to_dim"] = project_to_dim
        self.config.backbone_cfg["tune_llm"] = tune_llm
        self.config.backbone_cfg["tune_visual"] = tune_visual
        if "eagle_path" in self.config.backbone_cfg:
            del self.config.backbone_cfg["eagle_path"]
        self.config.backbone_cfg["select_layer"] = self.backbone.select_layer

    def update_action_head(self, config: GR00T_N1_5_Config, dit_num_layers: int, action_dim: int = None):
        del self.action_head
        if dit_num_layers != -1:
            config.action_head_cfg["diffusion_model_cfg"]["num_layers"] = dit_num_layers
            self.config.action_head_cfg["diffusion_model_cfg"]["num_layers"] = dit_num_layers
        
        # 支持自定义 action_dim（用于不同维度的数据集微调）
        if action_dim is not None:
            config.action_head_cfg["action_dim"] = action_dim
            config.action_head_cfg["max_action_dim"] = action_dim
            self.config.action_head_cfg["action_dim"] = action_dim
            self.config.action_head_cfg["max_action_dim"] = action_dim
            self.config.action_dim = action_dim
            self.action_dim = action_dim
            print(f"  -> Updated action_dim to {action_dim}")
        
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        self.action_head = FlowmatchingActionHead(action_head_cfg)
        print(f"\n================== update_action_head: dit_num_layers={dit_num_layers}, action_dim={self.action_dim} ====================\n")

    def enable_latent_alignment(self, M, hidden_dim):
        # QwenVL
        # sets the attribute directly in the instance's __dict__ without registering it in _modules
        # The model remains accessible via self.vl_embedding_model,
        # but the state dict will only include the parameters once
        # (under the backbone.eagle_model.vision_model. prefix).
        # object.__setattr__(self, 'vl_embedding_model', self.backbone.eagle_model.vision_model)
        # self.vl_embedding_model.requires_grad_(False)
        # self.vl_embedding_model.eval()

        self.future_tokens = nn.Embedding(M, hidden_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        # 2-layer MLP to decode predicted embeddings
        self.embedding_decode = SimpleMLP(hidden_dim, hidden_dim, 2048)

    # def enable_latent_alignment(self, M, hidden_dim): # GR00T
    #     # sets the attribute directly in the instance's __dict__ without registering it in _modules
    #     # The model remains accessible via self.vl_embedding_model,
    #     # but the state dict will only include the parameters once
    #     # (under the backbone.eagle_model.vision_model. prefix).
    #     object.__setattr__(self, 'vl_embedding_model', self.backbone.eagle_model.vision_model)
    #     self.vl_embedding_model.requires_grad_(False)
    #     self.vl_embedding_model.eval()
    #
    #     self.future_tokens = nn.Embedding(M, hidden_dim)
    #     nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
    #
    #     # 2-layer MLP to decode predicted embeddings
    #     self.embedding_decode = SimpleMLP(hidden_dim, hidden_dim, self.vl_embedding_model.config.hidden_size)

    def validate_inputs(self, inputs):
        # NOTE -- this should be handled internally by the model
        # however, doing that will likely be breaking changes -- so we'll need to do it after the deadline

        detected_error = False
        error_msg = ERROR_MSG
        if "action" in inputs:
            action = inputs["action"]
            type_ok = isinstance(action, torch.Tensor)
            shape_ok = (
                len(action.shape) == 3
                and action.shape[1] == self.action_horizon
                and action.shape[2] == self.action_dim
            )
            if not type_ok:
                error_msg += f"\n{action.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{action.shape=}"
                detected_error = True

        if "video" in inputs:
            video = inputs["video"]
            type_ok = isinstance(video, np.ndarray)
            dtype_ok = video.dtype == np.uint8
            shape_ok = len(video.shape) == 6 and video.shape[3] == N_COLOR_CHANNELS
            if not type_ok:
                error_msg += f"\n{type(video)=}"
                detected_error = True
            if not dtype_ok:
                error_msg += f"\n{video.dtype=}"
                detected_error = True
            if not shape_ok:
                error_msg += f"\n{video.shape=}"
                detected_error = True

        if detected_error:
            raise ValueError(error_msg)

    def validate_data(self, action_head_outputs, backbone_outputs, is_training):
        fail_backbone = (
            not isinstance(backbone_outputs, BatchFeature)
            or BACKBONE_FEATURE_KEY not in backbone_outputs
        )

        if fail_backbone:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(backbone_outputs, BatchFeature)=}"
            error_msg += f"\n{BACKBONE_FEATURE_KEY in backbone_outputs=}"
            error_msg += f"\n{backbone_outputs[BACKBONE_FEATURE_KEY].shape=}"
            raise ValueError(error_msg)

        fail_action_head = (not isinstance(action_head_outputs, BatchFeature)) or not (
            (
                LOSS_KEY in action_head_outputs and is_training
            )  # there might not be an action prediction during training
            or (
                ACTION_KEY in action_head_outputs
                and action_head_outputs[ACTION_KEY].shape[1] == self.action_horizon
                and action_head_outputs[ACTION_KEY].shape[2] == self.action_dim
            )
        )

        if fail_action_head:
            error_msg = ERROR_MSG
            error_msg += f"\n{isinstance(action_head_outputs, BatchFeature)=}"
            error_msg += f"\n{LOSS_KEY in action_head_outputs=}"
            error_msg += f"\n{action_head_outputs[ACTION_KEY].shape=}"
            error_msg += f"\n{self.action_horizon=}"
            error_msg += f"\n{self.action_dim=}"
            raise ValueError(error_msg)

    def vis_image(self, inputs):
        import matplotlib.pyplot as plt

        input_imgs = inputs['eagle_pixel_values']
        future_imgs = inputs['future_pixel_values']

        plt.figure(figsize=(8, 4))
        for i in range(2):
            # input image
            plt.subplot(2, 2, i*2 + 1)
            img = input_imgs[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow((img * 0.5 + 0.5).clip(0, 1))
            plt.axis('off')
            plt.title(f"Current {i}")

            # future image
            plt.subplot(2, 2, i*2 + 2)
            img = future_imgs[i].permute(1, 2, 0).cpu().numpy()
            plt.imshow((img * 0.5 + 0.5).clip(0, 1))
            plt.axis('off')
            plt.title(f"Future {i}")

        plt.tight_layout()
        plt.show()

    @torch.no_grad()
    def vl_embedding_forward(self, batch_size, inputs):
        future_vit_embeds = self.vl_embedding_model(
            pixel_values=inputs['future_pixel_values'], output_hidden_states=False, return_dict=True
        )

        if hasattr(future_vit_embeds, "last_hidden_state"):
            future_vit_embeds = future_vit_embeds.last_hidden_state
        else:
            raise NotImplementedError
        return future_vit_embeds

    def forward(
        self,
        inputs: dict,
    ) -> BatchFeature:

        batch_size = inputs['state'].shape[0]
        # batch, image_index, [3, 224, 224]

        if self.vl_embedding_model is not None:
            # self.vis_image(inputs)
            future_vit_embeds = self.vl_embedding_forward(batch_size, inputs)
            future_tokens_batch = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(backbone_inputs)

        if self.use_dino:
            vit_states = self.forward_dinov3(backbone_inputs['vit_inputs']) # [B, C, D]
            vit_states = vit_states.view(batch_size, -1, vit_states.shape[-1]).contiguous()

            backbone_features = backbone_outputs['backbone_features'] # [B, L, D]

            with torch.cuda.amp.autocast(dtype=torch.float32, enabled=True):
                fuse_features = self.vision_tokenizer(
                    vit_states.unsqueeze(1).unsqueeze(1)
                ).squeeze(1)
                fuse_features = self.vit_linear(fuse_features)

            concatenated_features = torch.cat((backbone_features, fuse_features), dim=1)
            vit_attention_mask = torch.ones((concatenated_features.shape[0], fuse_features.shape[1]), dtype=torch.long,
                                                device=concatenated_features.device)
            backbone_attention_mask = backbone_outputs['backbone_attention_mask']
            concatenated_attention_mask = torch.cat((backbone_attention_mask, vit_attention_mask), dim=1)

            backbone_outputs['backbone_features'] = concatenated_features
            backbone_outputs['backbone_attention_mask'] = concatenated_attention_mask

        if self.vl_embedding_model is not None:
            action_head_outputs = self.action_head(backbone_outputs, action_inputs, future_tokens=future_tokens_batch)

            predict_embedding = action_head_outputs.pop('predict_embedding')
            predict_embedding = self.embedding_decode(predict_embedding)
            cos_sim = F.cosine_similarity(predict_embedding, future_vit_embeds, dim=-1).mean()
            # map cos sim [-1, 1] -> [0, 2]
            latent_loss = 1 - cos_sim
            action_head_outputs['latent_loss'] = 0.2 * latent_loss
            flow_loss = action_head_outputs['loss']
            action_head_outputs['loss'] = flow_loss + action_head_outputs['latent_loss']
            action_head_outputs['flow_loss'] = flow_loss

        else:
            action_head_outputs = self.action_head(backbone_outputs, action_inputs)
        self.validate_data(action_head_outputs, backbone_outputs, is_training=True)
        return action_head_outputs


    def forward_dinov3(self, vit_inputs):
        outputs = self.dino_vit(vit_inputs)
        # [cls_token, 4 register_tokens, patch_embeddings： 14*14]
        last_hidden_state = outputs['last_hidden_state'][:, 5:, :].contiguous()
        # vit_states = self.vit_linear(last_hidden_state)
        return last_hidden_state

    def get_action(
        self,
        inputs: dict,
    ) -> BatchFeature:
        batch_size = inputs['state'].shape[0]

        if self.vl_embedding_model is not None:
            # self.vis_image(inputs)
            # future_vit_embeds = self.vl_embedding_forward(batch_size, inputs)
            future_tokens_batch = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)

        backbone_inputs, action_inputs = self.prepare_input(inputs)
        # Because the behavior of backbones remains the same for training and inference, we can use `forward` for backbones.
        backbone_outputs = self.backbone(backbone_inputs)

        if self.vl_embedding_model is not None:
            action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs, future_tokens=future_tokens_batch)
        else:
            action_head_outputs = self.action_head.get_action(backbone_outputs, action_inputs)

        self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        return action_head_outputs

    def prepare_input(self, inputs) -> Tuple[BatchFeature, BatchFeature]:
        # 只在第一个step打印，避免日志过多
        if not hasattr(self, '_first_input_logged'):
            if "action" in inputs:
                action = inputs["action"]
                if isinstance(action, torch.Tensor):
                    print(f"\n{'='*60}")
                    print(f"[Model.prepare_input] Action shape: {action.shape}")
                    print(f"Model action_horizon: {self.action_horizon}, action_dim: {self.action_dim}")
                    print(f"{'='*60}\n")
            self._first_input_logged = True

        self.validate_inputs(inputs)
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def to_device_with_maybe_dtype(x):
            # Only cast to self.compute_dtype if the tensor is floating
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.action_head.dtype)
            else:
                # Keep original dtype
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_maybe_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_maybe_dtype, action_inputs)
        return backbone_inputs, action_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        tune_visual = kwargs.pop("tune_visual", True)
        tune_llm = kwargs.pop("tune_llm", False)
        tune_projector = kwargs.pop("tune_projector", True)
        tune_diffusion_model = kwargs.pop("tune_diffusion_model", True)

        print(f"Loading pretrained dual brain from {pretrained_model_name_or_path}")
        print(f"Tune backbone vision tower: {tune_visual}")
        print(f"Tune backbone LLM: {tune_llm}")
        print(f"Tune action head projector: {tune_projector}")
        print(f"Tune action head DiT: {tune_diffusion_model}")

        # get the current model path being downloaded
        try:
            # NOTE(YL) This downloads the model to the local cache and returns the local path to the model
            # saved in ~/.cache/huggingface/hub/
            local_model_path = snapshot_download(pretrained_model_name_or_path, repo_type="model")
            # HFValidationError, RepositoryNotFoundError
        except (HFValidationError, RepositoryNotFoundError):
            print(
                f"Model not found or avail in the huggingface hub. Loading from local path: {pretrained_model_name_or_path}"
            )
            local_model_path = pretrained_model_name_or_path
        
        load_pretrained = kwargs.get("load_pretrained", False)
        kwargs.pop("load_pretrained", None)
        if kwargs.get('use_time_aware_action_head', False) and not load_pretrained:
            # 从config初始化, from_pretrained不会均值初始化自定义ActionHead的模型参数
            pretrained_config = GR00T_N1_5_Config.from_pretrained(local_model_path)
            pretrained_config.action_head_cfg['diffusion_model_cfg'] =  kwargs.pop('time_aware_action_head_config', None)
            pretrained_model = GR00T_N1_5(config=pretrained_config, local_model_path=local_model_path, **kwargs)
            print(f"================== VLA Load from config ====================")
        else:
            kwargs.pop("time_aware_action_head_config", None)
            pretrained_model = super().from_pretrained(
                local_model_path, local_model_path=local_model_path, **kwargs
            )
            print(f"================== VLA Load from pretrained ====================")

        pretrained_model.config.backbone_cfg["tune_llm"] = tune_llm
        pretrained_model.config.backbone_cfg["tune_visual"] = tune_visual
        pretrained_model.config.action_head_cfg["tune_projector"] = tune_projector
        pretrained_model.config.action_head_cfg["tune_diffusion_model"] = tune_diffusion_model

        pretrained_model.backbone.set_trainable_parameters(
            tune_visual=tune_visual, tune_llm=tune_llm
        )
        pretrained_model.action_head.set_trainable_parameters(
            tune_projector=tune_projector, tune_diffusion_model=tune_diffusion_model
        )
        return pretrained_model


# register
AutoConfig.register("gr00t_n1_5", GR00T_N1_5_Config)
AutoModel.register(GR00T_N1_5_Config, GR00T_N1_5)
