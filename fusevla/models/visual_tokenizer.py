import torch
from torch import einsum, nn
from einops import rearrange, reduce, repeat
from functools import wraps, partial


def _many(fn):
    @wraps(fn)
    def inner(tensors, pattern, **kwargs):
        return (fn(tensor, pattern, **kwargs) for tensor in tensors)
    return inner

rearrange_many = _many(rearrange)

def has_fn(model, fn_name):
    """Check if model has a function fn_name"""
    return callable(getattr(model, fn_name, None))


def exists(val):
    return val is not None


class VisionTokenizer(nn.Module):
    def __init__(self, dim_media, num_tokens_per_media):
        super().__init__()
        self.dim_media = dim_media
        self.num_tokens_per_media = num_tokens_per_media

def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )

class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat(
            (x, latents), dim=-2
        )  # TODO: Change the shape of vision attention mask according to this.
        if vision_attn_masks is not None:
            vision_attn_masks = torch.cat(
                (
                    vision_attn_masks,
                    torch.ones(
                        (latents.shape[0], latents.shape[-2]),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                ),
                dim=-1,
            )
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q, k, v = rearrange_many((q, k, v), "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = einsum("... i d, ... j d  -> ... i j", q, k)
        # Apply vision attention mask here.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
        if vision_attn_masks is not None:
            attn_bias = torch.zeros(
                (q.size(0), 1, 1, q.size(-2), k.size(-2)),
                dtype=q.dtype,
                device=q.device,
            )
            vision_attn_masks = repeat(
                vision_attn_masks, "b n -> b 1 1 l n", l=q.size(-2)
            )
            attn_bias.masked_fill_(vision_attn_masks.logical_not(), float("-inf"))
            sim += attn_bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        return self.to_out(out)


class PerceiverResampler(VisionTokenizer):
    def __init__(
        self,
        *,
        dim,
        dim_inner=None,
        depth=6,
        dim_head=96,
        heads=16,
        num_latents=128,
        max_num_media=None,
        max_num_frames=None,
        ff_mult=4,
    ):
        """
        Perceiver module which takes in image features and outputs image tokens.
        Args:
            dim (int): dimension of the incoming image features
            dim_inner (int, optional): final dimension to project the incoming image features to;
                also the final dimension of the outputted features. If None, no projection is used, and dim_inner = dim.
            depth (int, optional): number of layers. Defaults to 6.
            dim_head (int, optional): dimension of each head. Defaults to 64.
            heads (int, optional): number of heads. Defaults to 8.
            num_latents (int, optional): number of latent tokens to use in the Perceiver;
                also corresponds to number of tokens per sequence to output. Defaults to 64.
            max_num_media (int, optional): maximum number of media per sequence to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            max_num_frames (int, optional): maximum number of frames to input into the Perceiver
                and keep positional embeddings for. If None, no positional embeddings are used.
            ff_mult (int, optional): dimension multiplier for the feedforward network. Defaults to 4.
        """
        if dim_inner is not None:
            projection = nn.Linear(dim, dim_inner)
        else:
            projection = None
            dim_inner = dim
        super().__init__(dim_media=dim, num_tokens_per_media=num_latents)
        self.projection = projection
        self.latents = nn.Parameter(torch.randn(num_latents, dim))

        # positional embeddings
        self.frame_embs = (
            nn.Parameter(torch.randn(max_num_frames, dim))
            if exists(max_num_frames)
            else None
        )
        self.media_time_embs = (
            nn.Parameter(torch.randn(max_num_media, 1, dim))
            if exists(max_num_media)
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, vision_attn_masks=None):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
            vision_attn_masks (torch.Tensor): attention masks for padded visiont tokens (i.e., x)
                shape (b, v)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if exists(self.frame_embs):
            frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
            x = x + frame_embs
        x = rearrange(
            x, "b T F v d -> b T (F v) d"
        )  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = self.latents
        latents = repeat(latents, "n d -> b T n d", b=b, T=T)
        for attn, ff in self.layers:
            latents = attn(x, latents, vision_attn_masks) + latents
            latents = ff(latents) + latents

        if exists(self.projection):
            return self.projection(self.norm(latents))
        else:
            return self.norm(latents)

def model_test():
    model = PerceiverResampler(
        dim=2048,
        dim_inner=512,
        num_latents=128,
    )

    batch_size = 2
    num_media = 2
    num_frames = 1
    vlen = 256
    input_dim = 2048

    # 创建输入张量 (b, T, F, v, D)
    x = torch.randn(
        batch_size,
        num_media,
        num_frames,
        vlen,
        input_dim
    )

    # 可选：创建注意力掩码（处理填充）
    vision_attn_masks = torch.ones(batch_size, vlen)  # 全1掩码

    # 前向传播
    with torch.no_grad():
        output = model(x, vision_attn_masks)

    print(output.shape)  # torch.Size([2, 2, 128, 512])

# model_test()