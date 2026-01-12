# --------------------------------------------------------
# InternVL
# Copyright (c) 2023 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.layers import DropPath
from torch import nn
from torch.nn import init
from transformers.activations import ACT2FN
from transformers.modeling_outputs import (BaseModelOutput,
                                           BaseModelOutputWithPooling)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from einops import rearrange, repeat, reduce, pack, unpack
from .configuration_intern_vit import InternVisionConfig
import os
import numpy as np
import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

try:
    from .flash_attention import FlashAttention
    has_flash_attn = True
except:
    print('FlashAttention is not installed.')
    has_flash_attn = False


logger = logging.get_logger(__name__)


class InternRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


try:
    from apex.normalization import FusedRMSNorm

    InternRMSNorm = FusedRMSNorm  # noqa

    logger.info('Discovered apex.normalization.FusedRMSNorm - will use it instead of InternRMSNorm')
except ImportError:
    # using the normal InternRMSNorm
    pass
except Exception:
    logger.warning('discovered apex but it failed to load, falling back to InternRMSNorm')
    pass


NORM2FN = {
    'rms_norm': InternRMSNorm,
    'layer_norm': nn.LayerNorm,
}


class InternVisionEmbeddings(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(
            torch.randn(1, 1, self.embed_dim),
        )

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        self.position_embedding = nn.Parameter(torch.randn(1, self.num_positions, self.embed_dim))

    def _get_pos_embed(self, pos_embed, H, W):
        target_dtype = pos_embed.dtype
        pos_embed = pos_embed.float().reshape(
            1, self.image_size // self.patch_size, self.image_size // self.patch_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(H, W), mode='bicubic', align_corners=False).\
            reshape(1, -1, H * W).permute(0, 2, 1).to(target_dtype)
        return pos_embed

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(pixel_values)  # shape = [*, channel, width, height]
        batch_size, _, height, width = patch_embeds.shape
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1).to(target_dtype)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        position_embedding = torch.cat([
            self.position_embedding[:, :1, :],
            self._get_pos_embed(self.position_embedding[:, 1:, :], height, width)
        ], dim=1)
        embeddings = embeddings + position_embedding.to(target_dtype)
        return embeddings


class InternAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.use_flash_attn = config.use_flash_attn and has_flash_attn
        # self.use_flash_attn = False
        if config.use_flash_attn and not has_flash_attn:
            print('Warning: Flash Attention is not available, use_flash_attn is set to False.')
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f'embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:'
                f' {self.num_heads}).'
            )

        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=config.qkv_bias)
  
        self.attn_drop = nn.Dropout(config.attention_dropout)
        self.proj_drop = nn.Dropout(config.dropout)

        self.qk_normalization = config.qk_normalization

        if self.qk_normalization:
            self.q_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
            self.k_norm = InternRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

        if self.use_flash_attn:
            self.inner_attn = FlashAttention(attention_dropout=config.attention_dropout)
        self.proj = nn.Linear(self.embed_dim, self.embed_dim)
        # self.prefix_gate = torch.nn.Parameter(torch.zeros(1, self.num_heads, 1, 1))
        self.gate_proj = nn.Linear(self.embed_dim, self.embed_dim)


    def _naive_attn(self, x,text_embeds):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        target_dtype = self.qkv.weight.dtype

        # if text_embeds is not None:
        #     B,L,C = text_embeds.shape
        #     prefix_qkv = self.qkv(text_embeds.to(target_dtype)).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        #     prefix_q,prefix_k,prefix_v = prefix_qkv.unbind(0)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)

        attn = ((q * self.scale) @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2)

        # if text_embeds is not None:
        #     prefix_attn = ((q * self.scale) @ prefix_k.transpose(-2, -1))
        #     prefix_attn = prefix_attn.softmax(dim=-1)
        #     prefix_attn = self.attn_drop(prefix_attn)

        #     prefix_x = (prefix_attn @ prefix_v).transpose(1, 2)
        #     prefix_delta = self.prefix_gate.view(1, 1, -1, 1).tanh() * prefix_x
        #     x = x + prefix_delta
        x = x.reshape(B,N,C).contiguous()
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.num_heads)
        q, k, v = qkv.unbind(2)

        if self.qk_normalization:
            q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
            k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)


        context, attn_weights = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=False
        )

        outs = self.proj(rearrange(context, 'b s h d -> b s (h d)'))
        outs = self.proj_drop(outs)

        if need_weights:
            return outs, attn_weights
        return outs, None

    def forward(self, hidden_states, text_embeds=None, need_weights: bool = False) -> torch.Tensor:
        if not self.use_flash_attn:
            x, attn_weights = self._naive_attn(hidden_states, text_embeds)
            if not need_weights:
                attn_weights = None
            return x, attn_weights
        return self._flash_attn(hidden_states, need_weights=need_weights)


class InternMLP(nn.Module):
    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        self.act = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class InternVisionEncoderLayer(nn.Module):
    def __init__(self, config: InternVisionConfig, drop_path_rate: float):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.norm_type = config.norm_type

        self.attn = InternAttention(config)
        self.mlp = InternMLP(config)
        self.norm1 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        self.norm2 = NORM2FN[self.norm_type](self.embed_dim, eps=config.layer_norm_eps)
        
        self.ls1 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.ls2 = nn.Parameter(config.initializer_factor * torch.ones(self.embed_dim))
        self.drop_path1 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()


    def forward(
            self,
            hidden_states: torch.Tensor,
            text_embeds,
            output_attentions: bool = False

    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """  
        
        # hidden_states = hidden_states + self.drop_path1(self.attn(self.norm1(hidden_states),text_embeds) * self.ls1)

        # hidden_states = hidden_states + self.drop_path2(self.mlp(self.norm2(hidden_states)) * self.ls2)
        # 注意力层
        attn_output, attn_weights = self.attn(self.norm1(hidden_states), text_embeds, need_weights=output_attentions)
        hidden_states = hidden_states + self.drop_path1(attn_output * self.ls1)

        # MLP层
        mlp_output = self.mlp(self.norm2(hidden_states))
        hidden_states = hidden_states + self.drop_path2(mlp_output * self.ls2)
        
        return hidden_states, attn_weights
        
        # return hidden_states


class InternVisionEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`InternEncoderLayer`].

    Args:
        config (`InternConfig`):
            The corresponding vision configuration for the `InternEncoder`.
    """

    def __init__(self, config: InternVisionConfig):
        super().__init__()
        self.config = config
        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)]
        self.layers = nn.ModuleList([
            InternVisionEncoderLayer(config, dpr[idx]) for idx in range(config.num_hidden_layers)])
        self.gradient_checkpointing = True

    def forward(
            self,
            inputs_embeds,
            text_embeds,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Embedded representation of the inputs. Should be float, not int tokens.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, 'output_attentions', False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_states = () if output_hidden_states else None
        hidden_states = inputs_embeds
        all_attn_weights = () if output_attentions else None
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    encoder_layer,
                    hidden_states,
                    text_embeds
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    text_embeds,
                    output_attentions=output_attentions
                )

            hidden_states, attn_weights = layer_outputs
            if output_attentions:
                all_attn_weights = all_attn_weights + (attn_weights,)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)


        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attn_weights] if v is not None)
        return hidden_states, encoder_states, all_attn_weights


class InternVisionModel(PreTrainedModel):
    main_input_name = 'pixel_values'
    config_class = InternVisionConfig
    _no_split_modules = ['InternVisionEncoderLayer']

    def __init__(self, config: InternVisionConfig):
        super().__init__(config)
        self.config = config

        self.embeddings = InternVisionEmbeddings(config)
        self.encoder = InternVisionEncoder(config)

        self.llm_hidden_size = 2048
        self.union = nn.Sequential(
            nn.LayerNorm(self.llm_hidden_size),
            nn.Linear(self.llm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        

    def resize_pos_embeddings(self, old_size, new_size, patch_size):
        pos_emb = self.embeddings.position_embedding
        _, num_positions, embed_dim = pos_emb.shape
        cls_emb = pos_emb[:, :1, :]
        pos_emb = pos_emb[:, 1:, :].reshape(1, old_size // patch_size, old_size // patch_size, -1).permute(0, 3, 1, 2)
        pos_emb = F.interpolate(pos_emb.float(), size=new_size // patch_size, mode='bicubic', align_corners=False)
        pos_emb = pos_emb.to(cls_emb.dtype).reshape(1, embed_dim, -1).permute(0, 2, 1)
        pos_emb = torch.cat([cls_emb, pos_emb], dim=1)
        self.embeddings.position_embedding = nn.Parameter(pos_emb)
        self.embeddings.image_size = new_size
        logger.info('Resized position embeddings from {} to {}'.format(old_size, new_size))

    def get_input_embeddings(self):
        return self.embeddings

    def forward(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            text_embeds: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            pixel_embeds: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None and pixel_embeds is None:
            raise ValueError('You have to specify pixel_values or pixel_embeds')

        if pixel_embeds is not None:
            hidden_states = pixel_embeds
        else:
            if len(pixel_values.shape) == 4:
                hidden_states = self.embeddings(pixel_values)
                if text_embeds is not None:
                    batch_size, _, _ = hidden_states.shape
                    text_embeds = text_embeds.repeat(batch_size, 1, 1)
                    text_embeds = self.union(text_embeds)
                    hidden_states = torch.cat([hidden_states, text_embeds], dim=1)

            else:
                raise ValueError(f'wrong pixel_values size: {pixel_values.shape}')
        
        hidden_states,encoder_states, all_attn_weights = self.encoder(
            inputs_embeds=hidden_states,
            text_embeds=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]


        if not return_dict:
            return (last_hidden_state, pooled_output, hidden_states, all_attn_weights) + (encoder_states,)

        return last_hidden_state, pooled_output, hidden_states, all_attn_weights


def cls_attention_map(
        last_attn: torch.Tensor,
        pixel_values: torch.Tensor,
        patch_size: int,
) -> torch.Tensor:
    if last_attn is None:
        raise ValueError('last_attn is None. 请在 forward 时传 output_attentions=True。')
    if last_attn.dim() != 4:
        raise ValueError(f'Expect last_attn with shape [B, heads, T, T], got {tuple(last_attn.shape)}')

    grid_h = pixel_values.shape[-2] // patch_size
    grid_w = pixel_values.shape[-1] // patch_size
    num_img_tokens = 1 + grid_h * grid_w

    # 只取 CLS 对图像 patch token 的注意力（忽略后续可能拼接的文本 token）
    cls_to_patch = last_attn[:, :, 0, 1:num_img_tokens]  # [B, heads, num_patches]
    attn_map = cls_to_patch.mean(dim=1)  # [B, num_patches]
    attn_map = attn_map.reshape(attn_map.shape[0], grid_h, grid_w)

    min_v = attn_map.amin(dim=(-2, -1), keepdim=True)
    max_v = attn_map.amax(dim=(-2, -1), keepdim=True)
    attn_map = (attn_map - min_v) / (max_v - min_v + 1e-6)
    return attn_map


def attention_heatmap(
        attn_map: torch.Tensor,
        save_path: str,
        image: Optional[torch.Tensor] = None,
        alpha: float = 0.45,
        cmap: str = 'jet',
) -> None:

    if attn_map.dim() == 2:
        attn_map = attn_map.unsqueeze(0)
    if attn_map.dim() != 3:
        raise ValueError(f'Expect attn_map with shape [B,h,w] or [h,w], got {tuple(attn_map.shape)}')

    batch = attn_map.shape[0]
    for i in range(batch):
        heat = attn_map[i].detach().float().cpu().numpy()

        fig = plt.figure(figsize=(6, 6), dpi=200)
        ax = plt.gca()
        ax.axis('off')

        if image is not None:
            img = image
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img[i]
                if img.dim() == 3 and img.shape[0] == 3:
                    img = img.permute(1, 2, 0)
                img = img.detach().float().cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)

            heat_t = torch.from_numpy(heat).unsqueeze(0).unsqueeze(0)
            heat_up = F.interpolate(heat_t, size=img.shape[:2], mode='bilinear', align_corners=False)
            heat_up = heat_up.squeeze().numpy()
            ax.imshow(heat_up, cmap=cmap, alpha=alpha)
        else:
            ax.imshow(heat, cmap=cmap)

        out = save_path
        if batch > 1:
            root, ext = os.path.splitext(save_path)
            out = f'{root}_{i}{ext or ".png"}'
        plt.savefig(out, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
