import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from typing import Optional
from scripts.train_utils import info_model
from gr00t.model.action_head.ella import PerceiverAttentionBlock, Timesteps, TimestepEmbedding
from gr00t.model.action_head.flow_matching_action_head import DiT

Batch_Size = 2
Chunk_Size = 1 + 16
Action_Dim = 1536 # 32*48
VL_Len = 45

from diffusers import ConfigMixin, ModelMixin
from diffusers.configuration_utils import register_to_config

class TimestepEncoder(nn.Module):
    def __init__(self, embedding_dim, compute_dtype=torch.float32):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=1)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def forward(self, timesteps):
        dtype = next(self.parameters()).dtype
        timesteps_proj = self.time_proj(timesteps).to(dtype)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)
        return timesteps_emb


class TimeAwareActionHead(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
        time_channel: int = 320,
        time_embedding_dim: int = 768,
        time_out_dim: int = None,
        action_dim=1536,
        vl_input_dim=2048,
        heads: int = 16,
        layers: int = 16,
        output_dim=None,
        compute_dtype=torch.float32,
    ):
        super().__init__()
        self.vl_input_dim = vl_input_dim
        self.output_dim = output_dim

        # self.position = Timesteps(
        #     time_channel, flip_sin_to_cos=True, downscale_freq_shift=0
        # )
        # self.time_embedding = TimestepEmbedding(
        #     in_channels=time_channel,
        #     time_embed_dim=time_embedding_dim,
        #     act_fn='silu',
        #     out_dim=time_out_dim,
        # )

        # Timestep encoder
        self.timestep_encoder = TimestepEncoder(
            embedding_dim=time_embedding_dim, compute_dtype=self.config.compute_dtype
        )

        self.time_aware_linear = nn.Linear(
            time_embedding_dim, action_dim, bias=True
        )

        if self.vl_input_dim is not None:
            self.proj_in = nn.Linear(vl_input_dim, action_dim)

        self.perceiver_block = PerceiverAttentionBlock(
            action_dim, heads, time_embedding_dim=time_embedding_dim
        )

        self.perceiver_blocks = nn.Sequential(
            *[
                PerceiverAttentionBlock(
                    action_dim, heads, time_embedding_dim=time_embedding_dim
                )
                for _ in range(layers)
            ]
        )

        if self.output_dim is not None:
            self.proj_out = nn.Sequential(
                nn.Linear(action_dim, output_dim), nn.LayerNorm(output_dim)
            )

    def forward(
        self,
        latents: torch.Tensor,  # Shape: (B, T, D)
        visual_language_states: torch.Tensor,  # Shape: (B, S, D)
        timestep: Optional[torch.LongTensor] = None,
    ):
        # device = visual_language_states.device
        # dtype = visual_language_states.dtype
        # time_feature = self.position(timestep).to(device, dtype=dtype)
        # time_feature = time_feature.unsqueeze(1)
        # time_embedding = self.time_embedding(time_feature)

        time_embedding = self.timestep_encoder(timestep).unsqueeze(1)

        latents = latents + self.time_aware_linear(
            torch.nn.functional.silu(time_embedding)
        )

        if self.vl_input_dim is not None:
            visual_language_states = self.proj_in(visual_language_states)

        for l_block in self.perceiver_blocks:
            latents = l_block(
                x=visual_language_states,
                latents=latents,
                timestep_embedding=time_embedding,
            )

        if self.output_dim is not None:
            latents = self.proj_out(latents)

        return latents

def run_dit():
    sa_embs = torch.randn((Batch_Size, Chunk_Size, Action_Dim), device='cuda')
    vl_embs = torch.randn((Batch_Size, VL_Len, 2048), device='cuda')
    vl_attn_mask = torch.ones((Batch_Size, VL_Len), device='cuda')
    t_discretized = torch.tensor((100, 200), device='cuda')
    for i in range(100):

        action_output = action_head(
            latents=sa_embs,
            visual_language_states=vl_embs,
            timestep=t_discretized,
        )

        # print(model_output.shape)
        print(action_output.shape)

if __name__ == '__main__':
    dit_config = {'attention_head_dim': 48, 'cross_attention_dim': 2048, 'dropout': 0.2, 'final_dropout': True, 'interleave_self_attention': True, 'norm_type': 'ada_norm', 'num_attention_heads': 32, 'num_layers': 16, 'output_dim': 1024, 'positional_embeddings': None}
    dit = DiT(**dit_config).cuda()

    action_head = TimeAwareActionHead().cuda()

    info_model(dit)

    info_model(action_head)

    run_dit()

