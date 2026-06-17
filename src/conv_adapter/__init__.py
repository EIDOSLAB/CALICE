from functools import partial
from .layers import ConvWithAdapter, SubpelConvWithAdapter
import sys
import torch.nn as nn
import yaml

from custom_comp.zoo import models
import torch
from custom_comp.zoo.pretrained import load_pretrained


# from custom_comp.layers import BasicLayer, SwinTransformerBlock
from compressai.zoo import cheng2020_attn, cheng2020_anchor
from compressai.layers import ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample



def swish(x, beta=1.0):
    return x * torch.sigmoid(x * beta)

def get_conv_adapt_model(model, config, force_alpha = None):
    print('Get ConvAdapter model!')

    with open(config, 'r') as file:
        adapt_conf = yaml.safe_load(file)

    adapt_rank = adapt_conf['r'] # 8

    if force_alpha is None:
        adapt_alpha = adapt_conf['alpha'] # 16
    else:
        adapt_alpha = force_alpha

    adapt_residual = adapt_conf['adapt_residual'] # True
    adapt_conv1_residual = adapt_conf['conv1_residual'] # True
    adapt_conv2_residual = adapt_conf['conv2_residual'] # True

    # downsample
    adapt_residual_stride = adapt_conf['adapt_residual_stride'] # True
    adapt_conv1_residual_stride = adapt_conf['conv1_residual_stride'] # True
    adapt_conv2_residual_stride = adapt_conf['conv2_residual_stride'] # True

    # upsample
    adapt_residual_upsample = adapt_conf['adapt_residual_upsample'] # True
    adapt_subpel_conv_residual_upsample = adapt_conf['subpel_conv_residual_upsample'] # True
    adapt_conv_residual_upsample = adapt_conf['conv_residual_upsample'] # True

    adapt_attention = adapt_conf['adapt_attention'] #[True,True,True,True]
    adapt_conv_a_attention = adapt_conf['conv_a']
    adapt_conv_b_attention = adapt_conf['conv_b']

    adapt_act = adapt_conf['activation']

    divide_rank = adapt_conf['divide_rank']


    print('Configs:')
    print(f'r: {adapt_rank}')
    print(f'alpha: {adapt_alpha}')
    print(f'adapt Residual Block: {adapt_residual}')
    print(f'conv1_residual: {adapt_conv1_residual}')
    print(f'conv2_residual: {adapt_conv2_residual}')

    print(f'(Downsample) adapt Residual Block Stride: {adapt_residual_stride}')
    print(f'(Downsample) conv1_residual_stride: {adapt_conv1_residual_stride}')
    print(f'(Downsample) conv2_residual_stride: {adapt_conv2_residual_stride}')

    print(f'(Upsample) adapt Residual Block Upsample: {adapt_residual_upsample}')
    print(f'(Upsample) subpel_conv_residual_upsample: {adapt_subpel_conv_residual_upsample}')
    print(f'(Upsample) conv_residual_upsample: {adapt_conv_residual_upsample}')


    print(f'adapt Attention Block: {adapt_attention}')
    print(f'conv_a_attention: {adapt_conv_a_attention}')
    print(f'conv_b_attention: {adapt_conv_b_attention}')


    print(f'activation: {adapt_act}')

    if adapt_act == 'gelu':
        print('using gelu')
        act = nn.GELU()
    elif adapt_act == 'relu':
        print('using relu')
        act = nn.ReLU()
    elif adapt_act == 'swish':
        print('using swish')
        act = swish
    else:
        print('using identity')
        act = nn.Identity()

    assign_conv_adapter = partial(ConvWithAdapter, rank=adapt_rank, alpha=adapt_alpha, activation = act, divide_rank = divide_rank)
    if adapt_subpel_conv_residual_upsample:
        assign_subpel_conv_adapter = partial(SubpelConvWithAdapter, rank=adapt_rank, alpha=adapt_alpha, activation = act, divide_rank = divide_rank)
    
    for _,layer in enumerate(model.g_a): # encoder
        if isinstance(layer, ResidualBlock) and adapt_residual:
            if adapt_conv1_residual:
                layer.conv1 = assign_conv_adapter(layer.conv1)
            if adapt_conv2_residual:
                layer.conv2 = assign_conv_adapter(layer.conv2)
        elif isinstance(layer, ResidualBlockWithStride) and adapt_residual_stride:
            if adapt_conv1_residual_stride:
                layer.conv1 = assign_conv_adapter(layer.conv1)
            if adapt_conv2_residual_stride:
                layer.conv2 = assign_conv_adapter(layer.conv2)
        elif isinstance(layer, AttentionBlock) and adapt_attention:
            if adapt_conv_a_attention:
                for units in layer.conv_a:
                    units.conv[2] = assign_conv_adapter(units.conv[2])
            if adapt_conv_b_attention:
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = assign_conv_adapter(units.conv[2])

    for _,layer in enumerate(model.g_s): # decoder
        if isinstance(layer, ResidualBlock) and adapt_residual:
            if adapt_conv1_residual:
                layer.conv1 = assign_conv_adapter(layer.conv1)
            if adapt_conv2_residual:
                layer.conv2 = assign_conv_adapter(layer.conv2)
        elif isinstance(layer, ResidualBlockUpsample) and adapt_residual_upsample:
            if adapt_subpel_conv_residual_upsample:
                layer.subpel_conv = assign_subpel_conv_adapter(layer.subpel_conv)
            if adapt_conv_residual_upsample:
                layer.conv = assign_conv_adapter(layer.conv)
        elif isinstance(layer, AttentionBlock) and adapt_attention:
            if adapt_conv_a_attention:
                for units in layer.conv_a:
                    units.conv[2] = assign_conv_adapter(units.conv[2])
            if adapt_conv_b_attention:
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        units.conv[2] = assign_conv_adapter(units.conv[2])


    return model,adapt_conf





def change_model_alpha(model, new_alpha, config):
    # print('Change alpha of Conv Model')
    adapt_residual = config['adapt_residual'] 
    adapt_conv1_residual = config['conv1_residual'] 
    adapt_conv2_residual = config['conv2_residual'] 

    # downsample
    adapt_residual_stride = config['adapt_residual_stride'] 
    adapt_conv1_residual_stride = config['conv1_residual_stride'] 
    adapt_conv2_residual_stride = config['conv2_residual_stride'] 

    # upsample
    adapt_residual_upsample = config['adapt_residual_upsample'] 
    adapt_subpel_conv_residual_upsample = config['subpel_conv_residual_upsample'] 
    adapt_conv_residual_upsample = config['conv_residual_upsample'] 

    adapt_attention = config['adapt_attention'] 
    adapt_conv_a_attention = config['conv_a']
    adapt_conv_b_attention = config['conv_b']

    i = 0
    for _,layer in enumerate(model.g_a): # encoder
        if isinstance(layer, ResidualBlock) and adapt_residual:
            if adapt_conv1_residual:
                # print(layer.conv1.adapter.alpha)
                # i+=1
                layer.conv1.adapter.alpha = new_alpha
            if adapt_conv2_residual:
                # print(layer.conv2.adapter.alpha)
                # i+=1
                layer.conv2.adapter.alpha = new_alpha
        elif isinstance(layer, ResidualBlockWithStride) and adapt_residual_stride:
            if adapt_conv1_residual_stride:
                # print(layer.conv1.adapter.alpha)
                # i+=1
                layer.conv1.adapter.alpha = new_alpha
            if adapt_conv2_residual_stride:
                # print(layer.conv2.adapter.alpha)
                # i+=1
                layer.conv2.adapter.alpha = new_alpha
        elif isinstance(layer, AttentionBlock) and adapt_attention:
            if adapt_conv_a_attention:
                for units in layer.conv_a:
                    # print(units.conv[2].adapter.alpha)
                    # i+=1
                    units.conv[2].adapter.alpha = new_alpha
            if adapt_conv_b_attention:
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        # print(units.conv[2].adapter.alpha)
                        # i+=1
                        units.conv[2].adapter.alpha = new_alpha

    for _,layer in enumerate(model.g_s): # decoder
        if isinstance(layer, ResidualBlock) and adapt_residual:
            if adapt_conv1_residual:
                # print(layer.conv1.adapter.alpha)
                # i+=1
                layer.conv1.adapter.alpha = new_alpha
            if adapt_conv2_residual:
                # print(layer.conv2.adapter.alpha)
                # i+=1
                layer.conv2.adapter.alpha = new_alpha
        elif isinstance(layer, ResidualBlockUpsample) and adapt_residual_upsample:
            if adapt_subpel_conv_residual_upsample:
                # print(layer.subpel_conv.adapter.alpha)
                # i+=1
                layer.subpel_conv.adapter.alpha = new_alpha
            if adapt_conv_residual_upsample:
                # print(layer.conv.adapter.alpha)
                # i+=1
                layer.conv.adapter.alpha = new_alpha
        elif isinstance(layer, AttentionBlock) and adapt_attention:
            if adapt_conv_a_attention:
                for units in layer.conv_a:
                    # print(units.conv[2].adapter.alpha)
                    # i+=1
                    units.conv[2].adapter.alpha = new_alpha
            if adapt_conv_b_attention:
                for units in layer.conv_b:
                    if not isinstance(units, nn.Conv2d): # To avoid last 1x1 conv
                        # print(units.conv[2].adapter.alpha)
                        # i+=1
                        units.conv[2].adapter.alpha = new_alpha
        




