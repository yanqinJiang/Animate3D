from importlib import import_module
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import einsum, nn

from diffusers.utils import USE_PEFT_BACKEND
from diffusers.models.attention_processor import Attention

import xformers

from einops import rearrange, repeat
import math

from diffusers.models.embeddings import LabelEmbedding
from diffusers.models.resnet import AlphaBlender

from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from animatediff.models.embeddings import SinePositionalEncoding2D, LearnedPositionalEncoding2D 

class MVDreamXFormersAttnProcessor:
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None, num_views=4, num_frames=8):
        self.attention_op = attention_op
        self.num_views = num_views
        self.num_frames = num_frames

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        # MVDream: multi-view self-attention
        is_cross_attention = encoder_hidden_states is not None

        if not is_cross_attention:
            hidden_states = rearrange(hidden_states, "(b n f) l c -> (b f) (n l) c", n=self.num_views, f=self.num_frames).contiguous()

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # MVDream: multi-view self-attention back to normal shape
        if not is_cross_attention:
            hidden_states = rearrange(hidden_states, "(b f) (n l) c -> (b n f) l c", n=self.num_views, f=self.num_frames).contiguous()

        return hidden_states


class IPAdapterXFormersAttnProcessor(nn.Module):
    r"""
    Attention processor for Multiple IP-Adapater.

    Args:
        hidden_size (`int`):
            The hidden size of the attention layer.
        cross_attention_dim (`int`):
            The number of channels in the `encoder_hidden_states`.
        num_tokens (`int`, `Tuple[int]` or `List[int]`, defaults to `(4,)`):
            The context length of the image features.
        scale (`float` or List[`float`], defaults to 1.0):
            the weight scale of image prompt.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, num_tokens=(4,), scale=1.0,attention_op: Optional[Callable] = None, ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim

        if not isinstance(num_tokens, (tuple, list)):
            num_tokens = [num_tokens]
        self.num_tokens = num_tokens

        if not isinstance(scale, list):
            scale = [scale] * len(num_tokens)
        if len(scale) != len(num_tokens):
            raise ValueError("`scale` should be a list of integers with the same length as `num_tokens`.")
        self.scale = scale

        self.attention_op = attention_op
        
        self.to_k_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )
        self.to_v_ip = nn.ModuleList(
            [nn.Linear(cross_attention_dim, hidden_size, bias=False) for _ in range(len(num_tokens))]
        )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.FloatTensor] = None,
    ):
        residual = hidden_states

        # separate ip_hidden_states from encoder_hidden_states
        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`.This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to supress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # hidden_states = torch.bmm(attention_probs, value)
        # hidden_states = attn.batch_to_head_dim(hidden_states)

        # XFormers
        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)


        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, torch.Tensor) or ip_adapter_masks.ndim != 4:
                raise ValueError(
                    " ip_adapter_mask should be a tensor with shape [num_ip_adapter, 1, height, width]."
                    " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                )
            if len(ip_adapter_masks) != len(self.scale):
                raise ValueError(
                    f"Number of ip_adapter_masks ({len(ip_adapter_masks)}) must match number of IP-Adapters ({len(self.scale)})"
                )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        # for ip-adapter
        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            ip_key = to_k_ip(current_ip_hidden_states)
            ip_value = to_v_ip(current_ip_hidden_states)

            ip_key = attn.head_to_batch_dim(ip_key)
            ip_value = attn.head_to_batch_dim(ip_value)

            # ip_attention_probs = attn.get_attention_scores(query, ip_key, None)
            # current_ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
            # current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)
            
            # XFormers
            current_ip_hidden_states = xformers.ops.memory_efficient_attention(
                query, ip_key, ip_value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)
            current_ip_hidden_states = attn.batch_to_head_dim(current_ip_hidden_states)
        
            if mask is not None:
                mask_downsample = IPAdapterMaskProcessor.downsample(
                    mask, batch_size, current_ip_hidden_states.shape[1], current_ip_hidden_states.shape[2]
                )

                mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)

                current_ip_hidden_states = current_ip_hidden_states * mask_downsample

            hidden_states = hidden_states + scale * current_ip_hidden_states

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states



class MVDreamI2VXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(self, attention_op: Optional[Callable] = None, hidden_size=128, num_views=4, num_frames=8):
        super().__init__()

        self.attention_op = attention_op
        self.hidden_size = hidden_size
        self.num_views = num_views
        self.num_frames = num_frames

        self.to_q_i2v = nn.Linear(hidden_size, hidden_size, bias=False)
        self.to_out_i2v = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        # MVDream: multi-view self-attention
        is_cross_attention = encoder_hidden_states is not None

        if not is_cross_attention:
            hidden_states = rearrange(hidden_states, "(b n f) l c -> (b f) (n l) c", n=self.num_views, f=self.num_frames).contiguous()

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # I2V: store hidden_states, and prepare key and value
        # TODO: key -> i2v_key, value- > i2v_value, only the first frame !!
        origin_hidden_states = hidden_states.clone()

        first_key = rearrange(key, "(b f) l c -> b f l c", f=self.num_frames)
        first_key = first_key[:, 0:1].repeat_interleave(self.num_frames, dim=1)
        first_key = rearrange(first_key, "b f l c -> (b f) l c")
        i2v_key = attn.head_to_batch_dim(first_key).contiguous()

        first_value = rearrange(value, "(b f) l c -> b f l c", f=self.num_frames)
        first_value = first_value[:, 0:1].repeat_interleave(self.num_frames, dim=1)
        first_value = rearrange(first_value, "b f l c -> (b f) l c")
        i2v_value = attn.head_to_batch_dim(first_value).contiguous()
        # 
        # finish

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query, key, value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # I2V step
        # Note that the first frame is WITHOUT NOISE but WITH CAMERA EMBEDDING
        i2v_query = self.to_q_i2v(origin_hidden_states)
        i2v_query = attn.head_to_batch_dim(i2v_query).contiguous()

        i2v_hidden_states = xformers.ops.memory_efficient_attention(
            i2v_query, i2v_key, i2v_value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
        )
        i2v_hidden_states = i2v_hidden_states.to(i2v_query.dtype)
        i2v_hidden_states = attn.batch_to_head_dim(i2v_hidden_states)
        
        # I2V linear proj
        i2v_hidden_states = self.to_out_i2v(i2v_hidden_states)

        # sum
        hidden_states = hidden_states + i2v_hidden_states 

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # MVDream: multi-view self-attention back to normal shape
        if not is_cross_attention:
            hidden_states = rearrange(hidden_states, "(b f) (n l) c -> (b n f) l c", n=self.num_views, f=self.num_frames).contiguous()

        return hidden_states


class SpatioTemporalI2VXFormersAttnProcessor(nn.Module):
    r"""
    Processor for implementing memory efficient attention using xFormers.

    Args:
        attention_op (`Callable`, *optional*, defaults to `None`):
            The base
            [operator](https://facebookresearch.github.io/xformers/components/ops.html#xformers.ops.AttentionOpBase) to
            use as the attention operator. It is recommended to set to `None`, and allow xFormers to choose the best
            operator.
    """

    def __init__(
        self, 
        attention_op: Optional[Callable] = None, 
        hidden_size=128, 
        feature_size=64, 
        num_views=4, 
        num_frames=16,
        spatial_attn=None,
        image_attn=None,
        use_alpha_blender=False,
    ):
        super().__init__()

        self.attention_op = attention_op

        self.hidden_size = hidden_size
        self.feature_size = feature_size
        # self.batch_size = batch_size
        self.num_views = num_views
        self.num_frames = num_frames
        
        self.use_spatial_attn = spatial_attn.enabled
        self.use_spatial_encoding = spatial_attn.attn_cfg.use_spatial_encoding
        self.use_camera_encoding = spatial_attn.attn_cfg.use_camera_encoding
        self.spatial_encoding_type = spatial_attn.attn_cfg.spatial_encoding_type
        self.camera_encoding_type = spatial_attn.attn_cfg.camera_encoding_type
        
        self.use_image_attn = image_attn.enabled
        
        if self.use_spatial_attn:
            self.to_q_sp = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_k_sp = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_sp = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_sp = nn.Linear(hidden_size, hidden_size, bias=True)

            if self.use_spatial_encoding:
                # restore time embedding
                self.time_pos_embed = SinusoidalPositionalEmbedding(embed_dim=hidden_size, max_seq_length = 32)

                # use spatial_encoding
                if self.spatial_encoding_type == "sinusoid":
                    self.spatial_pos_embed = SinePositionalEncoding2D(hidden_size//2, normalize=True,)
                elif self.spatial_encoding_type == "learnable":
                    self.spatial_pos_embed = LearnedPositionalEncoding2D(hidden_size//2, row_num_embed=self.feature_size, col_num_embed=self.feature_size)
                else:
                    raise ValueError(f"Spatial encoding type {spatial_encoding_type} is not supported yet!")
            if self.use_camera_encoding:
                # restore time embedding
                self.time_pos_embed = SinusoidalPositionalEmbedding(embed_dim=hidden_size, max_seq_length = 32)
                if self.camera_encoding_type == "learnable":
                    self.camera_embed = LabelEmbedding(num_views, hidden_size, 0.)
                elif self.camera_encoding_type == "sinusoid":
                    self.camera_embed = SinusoidalPositionalEmbedding(embed_dim=hidden_size, max_seq_length = num_views)

        if self.use_image_attn:
            self.to_q_i2v = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_k_i2v = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_v_i2v = nn.Linear(hidden_size, hidden_size, bias=False)
            self.to_out_i2v = nn.Linear(hidden_size, hidden_size, bias=True)

        self.use_alpha_blender = use_alpha_blender
        num_attn = 1 # temporal attn
        if self.use_spatial_attn:
            num_attn += 1
        if self.use_image_attn:
            num_attn += 1
        
        if not self.use_alpha_blender:
            if self.use_spatial_attn:
                # zero init
                for param in self.to_out_sp.parameters():
                    nn.init.zeros_(param)
            if self.use_image_attn:
                # zero init
                for param in self.to_out_i2v.parameters():
                    nn.init.zeros_(param)
        elif num_attn == 2:
            self.alpha_blender = AlphaBlender(alpha=0.0, merge_strategy="learned")
        elif num_attn == 3:
            self.alpha_blender = SoftmaxAlphaBlender(alphas=[0.0, 0.0, 0.0])
            
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:

        # sample in is "(b n h w) f c"
        
        # Spatial
        if self.use_spatial_attn:
            seq_length = self.num_views * self.feature_size * self.feature_size
            spatial_hidden_states = rearrange(hidden_states, "(b l) f c -> (b f) l c", l=seq_length).contiguous()
            # resotre time positional embedding and use spatial positional embedding
            if self.use_spatial_encoding and self.spatial_pos_embed is not None:
                # hidden_states = self.time_pos_embed(hidden_states)
                spatial_hidden_states = rearrange(spatial_hidden_states, "b (n h w) c -> (b n) c h w", n=self.num_views, h=self.feature_size, w=self.feature_size).contiguous()
                spatial_hidden_states = self.spatial_pos_embed(spatial_hidden_states)
                spatial_hidden_states = rearrange(spatial_hidden_states, "(b n) c h w -> b (n h w) c", n=self.num_views)
                
            if self.use_camera_encoding and self.camera_embed is not None:
                if self.camera_encoding_type == "learnable":
                    spatial_hidden_states = rearrange(spatial_hidden_states, "b (n h w) c -> n c (b h w)", n=self.num_views, h=self.feature_size, w=self.feature_size).contiguous()
                    camera_index = torch.arange(self.num_views).long().cuda()
                    camera_embed = self.camera_embed(camera_index)
                    spatial_hidden_states = spatial_hidden_states + camera_embed.unsqueeze(-1) # add
                    spatial_hidden_states = rearrange(spatial_hidden_states, "n c (b h w) -> b (n h w) c", h=self.feature_size, w=self.feature_size)
                elif self.camera_encoding_type == "sinusoid":
                    spatial_hidden_states = rearrange(spatial_hidden_states, "b (n h w) c -> (b h w) n c", n=self.num_views, h=self.feature_size, w=self.feature_size).contiguous()
                    spatial_hidden_states = self.camera_embed(spatial_hidden_states)
                    spatial_hidden_states = rearrange(spatial_hidden_states, "(b h w) n c -> b (n h w) c", n=self.num_views, h=self.feature_size, w=self.feature_size).contiguous()

        # Image
        if self.use_image_attn:
            seq_length = self.feature_size * self.feature_size
            image_hidden_states = rearrange(hidden_states, "(b l) f c -> (b f) l c", l=seq_length).contiguous()
        
        # restore temporal encoding
        if self.use_spatial_attn and (self.use_spatial_encoding or self.use_camera_encoding):
            hidden_states = self.time_pos_embed(hidden_states)

        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, key_tokens, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(attention_mask, key_tokens, batch_size)
        if attention_mask is not None:
            # expand our mask's singleton query_tokens dimension:
            #   [batch*heads,            1, key_tokens] ->
            #   [batch*heads, query_tokens, key_tokens]
            # so that it can be added as a bias onto the attention scores that xformers computes:
            #   [batch*heads, query_tokens, key_tokens]
            # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
            _, query_tokens, _ = hidden_states.shape
            attention_mask = attention_mask.expand(-1, query_tokens, -1)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        ## Temporal: Animatediff
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        ## Spatial
        if self.use_spatial_attn:
            spatial_query = self.to_q_sp(spatial_hidden_states)

            spatial_encoder_hidden_states = spatial_hidden_states

            spatial_key = self.to_k_sp(spatial_encoder_hidden_states)
            spatial_value = self.to_v_sp(spatial_encoder_hidden_states)

            spatial_query = attn.head_to_batch_dim(spatial_query).contiguous()
            spatial_key = attn.head_to_batch_dim(spatial_key).contiguous()
            spatial_value = attn.head_to_batch_dim(spatial_value).contiguous()

            spatial_hidden_states = xformers.ops.memory_efficient_attention(
                spatial_query, spatial_key, spatial_value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            spatial_hidden_states = spatial_hidden_states.to(spatial_query.dtype)
            spatial_hidden_states = attn.batch_to_head_dim(spatial_hidden_states)
            # attention_probs = attn.get_attention_scores(query, key, attention_mask)
            # hidden_states = torch.bmm(attention_probs, value)
            # hidden_states = attn.batch_to_head_dim(hidden_states)

            # linear proj
            spatial_hidden_states = self.to_out_sp(spatial_hidden_states)
            # dropout
            # hidden_states = attn.to_out[1](hidden_states)
            spatial_hidden_states = rearrange(spatial_hidden_states, "(b f) l c -> (b l) f c", f=self.num_frames).contiguous()
        
        ## Image
        if self.use_image_attn:
            image_query = self.to_q_i2v(image_hidden_states)
    
            image_encoder_hidden_states = image_hidden_states
            image_encoder_hidden_states = rearrange(image_encoder_hidden_states, "(b f) l c -> b f l c", f=self.num_frames).contiguous()
            image_encoder_hidden_states = image_encoder_hidden_states[:, 0] # only first frame
            # image_encoder_hidden_states = rearrange(image_encoder_hidden_states, "b f l c -> (b f) l c")
            
            image_key = self.to_k_i2v(image_encoder_hidden_states)
            image_value = self.to_v_i2v(image_encoder_hidden_states)
            image_key = image_key.unsqueeze(1).repeat_interleave(self.num_frames, dim=1)
            image_value = image_value.unsqueeze(1).repeat_interleave(self.num_frames, dim=1)
            image_key = rearrange(image_key, "b f l c -> (b f) l c")
            image_value = rearrange(image_value, "b f l c -> (b f) l c")
            
            image_query = attn.head_to_batch_dim(image_query).contiguous()
            image_key = attn.head_to_batch_dim(image_key).contiguous()
            image_value = attn.head_to_batch_dim(image_value).contiguous()

            image_hidden_states = xformers.ops.memory_efficient_attention(
                image_query, image_key, image_value, attn_bias=attention_mask, op=self.attention_op, scale=attn.scale
            )
            image_hidden_states = image_hidden_states.to(image_query.dtype)
            image_hidden_states = attn.batch_to_head_dim(image_hidden_states)

            image_hidden_states = self.to_out_i2v(image_hidden_states)
            image_hidden_states = rearrange(image_hidden_states, "(b f) l c -> (b l) f c", f=self.num_frames).contiguous()

        if not self.use_alpha_blender:
            # merge Spatial and Temporal and etc.
            if self.use_spatial_attn:
                hidden_states = hidden_states + spatial_hidden_states
            if self.use_image_attn:
                hidden_states = hidden_states + image_hidden_states

        else:
            if self.use_spatial_attn and (not self.use_image_attn):
                hidden_states = self.alpha_blender(spatial_hidden_states, hidden_states)
            elif self.use_image_attn and (not self.use_spatial_attn):
                hidden_states = self.alpha_blender(image_hidden_states, hidden_states)
            else:
                hidden_states = self.alpha_blender(spatial_hidden_states, hidden_states, image_hidden_states)
        
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


# misc
class SoftmaxAlphaBlender(nn.Module):
    def __init__(self, alphas):
        super().__init__()
        
        self.register_parameter("mix_factor", torch.nn.Parameter(torch.Tensor(alphas)))

    def get_alpha(self):

        alpha = torch.softmax(self.mix_factor, dim=0)

        return alpha
    
    def forward(self, x_spatial, x_temporal, x_image):

        alpha = self.get_alpha()
        x = x_spatial * alpha[0] + x_temporal * alpha[1] + x_image * alpha[2]

        return x