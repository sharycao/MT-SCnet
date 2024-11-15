# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
# from . import vit_seg_configs as configs
# from .vit_seg_modeling_resnet_skip import ResNetV2
import vit_seg_configs as configs
from vit_seg_modeling_resnet_skip import ResNetV2
import torch.nn.functional as F
from einops import rearrange
import torchvision.ops
import QFormer

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}



class spe_spa_fusion(nn.Module):
    def __init__(self, channel):
        super(spe_spa_fusion,self).__init__()
        self.channel = channel
        self.att = nn.Tanh()
        self.apsilon = 1e-12
        self.avg = nn.AdaptiveAvgPool2d((1,1))


    def forward(self, res, spectral, spatial):
        spe_spa = spectral * spatial#光谱全局和空间全局的突出特征
        res_spectral = res * spectral#光谱全局和残差之间的突出特征
        res_spatial = res * spatial#空间全局和残差之间的突出特征
        att = self.att(spe_spa+res_spectral+res_spatial)+1
        return att



class Attention_spatial(nn.Module):
    def __init__(self, config, vis):
        super(Attention_spatial, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, x_dennes=None):

        if x_dennes != None:
            mixed_query_layer = self.query(hidden_states + x_dennes)
            mixed_key_layer = self.key(hidden_states + x_dennes)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)#4,16,768,16
        key_layer = self.transpose_for_scores(mixed_key_layer)#4,16,768,16
        value_layer = self.transpose_for_scores(mixed_value_layer)#4,16,768,16

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#4,16,768,16
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)#4,16,768,768

        context_layer = torch.matmul(attention_probs, value_layer)#4,16,768,16
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#4,768,12,21
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)#4,768,256
        return attention_output, weights



class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.patch_size**2 / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.patch_size**2, self.all_head_size)
        self.key = Linear(config.patch_size**2, self.all_head_size)
        self.value = Linear(config.patch_size**2, self.all_head_size)

        self.out = Linear(config.patch_size**2, config.patch_size**2)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, x_dennes=None):
        if x_dennes != None:
            mixed_query_layer = self.query(hidden_states + x_dennes)
            mixed_key_layer = self.key(hidden_states + x_dennes)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)#4,16,768,16
        key_layer = self.transpose_for_scores(mixed_key_layer)#4,16,768,16
        value_layer = self.transpose_for_scores(mixed_value_layer)#4,16,768,16

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)#4,16,768,16
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)#4,16,768,768

        context_layer = torch.matmul(attention_probs, value_layer)#4,16,768,16
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()#4,768,12,21
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)#4,768,256
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.patch_size**2, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.patch_size**2)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class attention_fusion(nn.Module):
    def __init__(self, chan_nums):
        super(attention_fusion, self).__init__()
        self.tanh = nn.Tanh()
        self.chan_nums = chan_nums

    def forward(self,fine, corse):
        fine_x_1 , fine_x_2 = torch.chunk(fine, 2, dim=1)
        corse_x_1, corse_x_2 = torch.chunk(corse, 2, dim=1)
        f_1 = fine_x_1 + corse_x_1
        fm_1 = self.tanh(f_1 * fine_x_2) + 1
        cm_2 = self.tanh(f_1 * corse_x_2) + 1
        fm_1 = fine_x_2 + fm_1
        cm_2 = corse_x_2 + cm_2
        fusion = (self.tanh(f_1 * fm_1 * cm_2) + 1) + f_1
        out = torch.cat((fusion, fm_1+cm_2),dim=1)
        return out

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None#是否使用ResNet
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            # fine
            grid_size_f = config.patches["grid"]#16
            patch_size_f = (img_size[0] // 16 // grid_size_f[0], img_size[1] // 16 // grid_size_f[1])#2
            patch_size_real_f = (patch_size_f[0] * 16, patch_size_f[1] * 16)#32
            n_patches_f = (img_size[0] // patch_size_real_f[0]) * (img_size[1] // patch_size_real_f[1])#256

            # corse
            grid_size_c = config.patches["grid"]
            patch_size_c = (img_size[0] * 2.5 // 16 // (grid_size_c[0]), img_size[1] * 2.5 // 16 // (grid_size_c[1]))#4
            patch_size_real_c = (patch_size_c[0] * 16, patch_size_c[1] * 16)#64
            n_patches_c = (img_size[0] * 2.5 // patch_size_real_c[0]) * (img_size[1] * 2.5 // patch_size_real_c[1])#256
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        # fine
        self.patch_embeddings_f = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=2,#2
                                       stride=2)
        # corse
        self.patch_embeddings_c = Conv2d(in_channels=in_channels,
                                         out_channels=config.hidden_size,
                                         kernel_size=4,#4
                                         stride=4)
        # self.pad_3 = nn.ReflectionPad2d(padding=(8, 8, 8, 8))
        self.pad = nn.ReflectionPad2d(padding=(16, 16, 16, 16))
        self.position_embeddings_f = nn.Parameter(torch.zeros(1, config.hidden_size, n_patches_f ))

        # self.position_embeddings_c = nn.Parameter(torch.zeros(1, config.hidden_size, n_patches_c))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        # 注意力融合
        self.attention_fusion = attention_fusion(768)

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)#4，1024，32，32
        else:
            features = None
        # x_f = self.pad_3(x)
        x_f = self.patch_embeddings_f(x) #4，768，16，16 # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x_c = self.pad(x)
        x_c = self.patch_embeddings_c(x_c)
        x_fusion = self.attention_fusion(x_f, x_c)
        x_fusion = x_fusion.flatten(2)
        embeddings_f = x_fusion + self.position_embeddings_f
        embeddings_f = self.dropout(embeddings_f)
        return embeddings_f, features

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.patch_size**2
        self.attention_norm = LayerNorm(config.patch_size**2, eps=1e-6)
        self.ffn_norm = LayerNorm(config.patch_size**2, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)
        self.attn_spatial = Attention_spatial(config, vis)
        self.fusion_att = spe_spa_fusion(768)

    def forward(self, x, layers):
        if len(layers) > 1:
            x_dennse = 0
            for i in range(len(layers)):
                x_dennse = x_dennse + layers[i]

            h = x
            x = self.attention_norm(x)
            x_spectral, weights = self.attn(x, x_dennse)

            x_spatial = rearrange(x, 'b n d-> b d n')  # 4,256,768
            x_dennse_c = rearrange(x_dennse, 'b n d-> b d n')  # 4,256,768
            x_spatial, weights = self.attn_spatial(x_spatial, x_dennse_c)
            x_spatial = rearrange(x_spatial, 'b d n -> b n d')
            att_fusion = self.fusion_att(h, x_spectral, x_spatial)
            x =  h + att_fusion

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
        else:
            h = x
            x = self.attention_norm(x)
            x_spectral, weights = self.attn(x)

            x_spatial = rearrange(x, 'b n d-> b d n')  # 4,256,768
            x_spatial, weights = self.attn_spatial(x_spatial)
            x_spatial= rearrange(x_spatial, 'b d n -> b n d')
            att_fusion = self.fusion_att(h, x_spectral, x_spatial)
            x = h + att_fusion

            h = x
            x = self.ffn_norm(x)
            x = self.ffn(x)
            x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.patch_size**2, eps=1e-6)
        self.encoder_norm_spatial = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))
        self.layer_spatial =  nn.ModuleList()

    def forward(self, hidden_states):
        hidden_states_spatial = hidden_states
        attn_weights = []
        layers = []
        layers.append(hidden_states)
        for i, layer_block in enumerate(self.layer):
            hidden_states, weights = layer_block(hidden_states, layers=layers)
            layers.append(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)#4,768,256
        # encoded_spatial = rearrange(hidden_states_spatial, 'b n d -> b d n')#4,256,768
        # for i, layer_block_spatial in enumerate(self.layer_spatial):
        #     encoded_spatial = layer_block_spatial(encoded_spatial)
        # encoded_spatial = self.encoder_norm_spatial(encoded_spatial)
        # encoded_spatial = rearrange(encoded_spatial, 'b d n -> b n d') + encoded

        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output_fine, features = self.embeddings(input_ids) # 4,768,256
        encoded, attn_weights = self.encoder(embedding_output_fine)  # (B, n_patch, hidden)

        return encoded, attn_weights, features

class DeformableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1,dilation=1,bias=False):
        super(DeformableConv2d, self).__init__()

        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        self.dilation = dilation

        self.offset_conv = nn.Conv2d(in_channels,2 * kernel_size[0] * kernel_size[1],kernel_size=kernel_size,
                                     stride=stride,padding=self.padding,dilation=self.dilation,bias=True)

        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)

        self.modulator_conv = nn.Conv2d(in_channels,1 * kernel_size[0] * kernel_size[1],kernel_size=kernel_size,
                                        stride=stride,padding=self.padding,dilation=self.dilation,bias=True)

        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)

        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size,
                                      stride=stride,padding=self.padding,dilation=self.dilation,bias=bias)

    def forward(self, x):
        # h, w = x.shape[2:]
        # max_offset = max(h, w)/4.

        offset = self.offset_conv(x)  # .clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        # op = (n - (k * d - 1) + 2p / s)
        x = torchvision.ops.deform_conv2d(input=x,offset=offset,weight=self.regular_conv.weight,bias=self.regular_conv.bias,
                                          padding=self.padding,mask=modulator,stride=self.stride,dilation=self.dilation)
        return x

class Conv2dReLU(nn.Sequential):
    def __init__(self,in_channels,out_channels,kernel_size,padding=0,stride=1,use_batchnorm=True,):
        conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride=stride,padding=padding,bias=not (use_batchnorm),)
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels=0,use_batchnorm=True,):
        super().__init__()
        # self.conv1 = Conv2dReLU(in_channels + skip_channels,out_channels,kernel_size=3,padding=1,use_batchnorm=use_batchnorm,)
        # self.conv2 = Conv2dReLU(out_channels,out_channels,kernel_size=3,padding=1,use_batchnorm=use_batchnorm,)
        self.conv1 = DeformableConv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DeformableConv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # if skip_channels != 0:
        #     self.high_low_fusion = RepBlock(in_channels + skip_channels, skip_channels,channelAttention_reduce=4)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # x = self.high_low_fusion(skip,x)
            x = torch.cat([x, skip], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)
        self.delinear = nn.Linear(in_features=self.config.hidden_size,out_features=self.config.hidden_size * 4)
        self.deconv = nn.ConvTranspose2d(self.config.hidden_size, self.config.hidden_size, kernel_size=(3, 3), stride=2, padding=1,
                                     output_padding=1, dilation=1)
        # self.spatial_trans = spatial_trans()
        self.conv1x1_down = nn.Conv2d(in_channels=config.hidden_size,out_channels=256, kernel_size=1,padding=0)
        self.conv1x1_up = nn.Conv2d(in_channels=256, out_channels=config.hidden_size, kernel_size=1, padding=0)
    def forward(self, hidden_states, features=None):

        B,hidden, n_patch = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states
        x = x.contiguous().view(B, hidden, h, w)
        x = self.deconv(x)#4,768,32,32
        # x = self.spatial_trans(features[2],features[1],features[0],x)
        # x = x.permute(0,2,1).contiguous().view(B, 768, 32, 32)#256,32,32
        # x_downsample = self.conv1x1_down(x) + x_q
        # x_upsample = self.conv1x1_up(x_q)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        # if x.size()[1] == 1:
        #     x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),

    'R50-ViT-B_16': configs.get_r50_b16_config(),

    'testing': configs.get_testing(),
}
if __name__ == '__main__':
    rgb = torch.randn(4, 2, 512, 512)
    from thop import profile
    from thop import clever_format
    from ptflops import get_model_complexity_info
    net = VisionTransformer(config=CONFIGS['R50-ViT-B_16'],
                            img_size=512,
                            num_classes=2,
                            zero_head=False,
                            vis=False)

    out = net(rgb)
    flops, params = profile(net, inputs=(rgb,))
    flops, params = clever_format([flops, params], '%.3f')

    print(out.shape)
    print(f"运算量：{flops}, 参数量：{params}")

    # print(f"模型 FLOPs: {macs}")
    # print(f"模型参数量: {params}")