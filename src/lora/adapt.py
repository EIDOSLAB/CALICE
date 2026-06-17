import torch
from functools import partial
from .layers import LinearWithLoRA
import sys
import torch.nn as nn
import yaml

from conv_adapter.layers import ConvWithAdapter, SubpelConvWithAdapter

from custom_comp.zoo import models
from .merge import get_merged_lora
import torch
from custom_comp.zoo.pretrained import load_pretrained


from custom_comp.layers import BasicLayer, SwinTransformerBlock # TODO fix this
# from stf_qvref import BasicLayer, SwinTransformerBlock
from compressai.zoo import cheng2020_attn


import sys



def swish(x, beta=1.0):
    return x * torch.sigmoid(x * beta)

def get_lora_model(model, config, force_alpha = None):
    print('Get LoRA model!')

    with open(config, 'r') as file:
        lora_conf = yaml.safe_load(file)

    lora_r = lora_conf['r'] # 8

    if force_alpha is None:
        lora_alpha = lora_conf['alpha'] # 16
    else:
        lora_alpha = force_alpha

    lora_mlp_fc1 = lora_conf['mlp_fc1'] # True
    lora_mlp_fc2 = lora_conf['mlp_fc2'] # True

    lora_encoder = lora_conf['encoder'] # [True,True,True,True]
    lora_decoder = lora_conf['decoder'] #[True,True,True,True]

    lora_act = lora_conf['activation'] # [True,True,True,True]
    divide_rank = lora_conf['divide_rank'] # True 

    adapt_hyp = lora_conf['adapt_hyp'] # True 

    

    print('Configs:')
    print(f'r: {lora_r}')
    print(f'alpha: {lora_alpha}')
    print(f'mlp_fc1: {lora_mlp_fc1}')
    print(f'mlp_fc2: {lora_mlp_fc2}')
    print(f'encoder: {lora_encoder}')
    print(f'activation: {lora_decoder}')
    print(f'adapt_hyp: {adapt_hyp}')

    if lora_act == 'gelu':
        print('using gelu')
        act = nn.GELU()
    elif lora_act == 'relu':
        print('using relu')
        act = nn.ReLU()
    elif lora_act == 'swish':
        print('using swish')
        act = swish
    else:
        print('using identity')
        act = nn.Identity()

    assign_lora = partial(LinearWithLoRA, rank=lora_r, alpha=lora_alpha, activation = act, divide_rank = divide_rank)

    for i,layer in enumerate(model.layers): # encoder
        print('encoder')
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            print('encoder basic layer')

            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    # weigths.append(swin_block.mlp.fc1.weight)
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)


    if adapt_hyp:
        assign_conv_adapter = partial(ConvWithAdapter, rank=lora_r, alpha=lora_alpha, activation = act, divide_rank = divide_rank)
        assign_subpel_conv_adapter = partial(SubpelConvWithAdapter, rank=lora_r, alpha=lora_alpha, activation = act, divide_rank = divide_rank)

        for i,layer in enumerate(model.h_a):
            if isinstance(layer, nn.Conv2d):
                model.h_a[i] = assign_conv_adapter(model.h_a[i])

        for i,layer in enumerate(model.h_mean_s):
            if isinstance(layer, nn.Conv2d):
                model.h_mean_s[i] = assign_conv_adapter(model.h_mean_s[i])
            elif isinstance(layer, nn.Sequential):
                model.h_mean_s[i] = assign_subpel_conv_adapter(model.h_mean_s[i])

        for i,layer in enumerate(model.h_scale_s):
            if isinstance(layer, nn.Conv2d):
                model.h_scale_s[i] = assign_conv_adapter(model.h_scale_s[i])
            elif isinstance(layer, nn.Sequential):
                model.h_scale_s[i] = assign_subpel_conv_adapter(model.h_scale_s[i])

    return model,lora_conf



def get_vanilla_finetuned_model(model):
    print('Get Vanilla FineTuned model!')

    lora_mlp_fc1 = True
    lora_mlp_fc2 = True

    lora_encoder = [True,True,True,True]
    lora_decoder = [True,True,True,True]

    print('Configs:')
    print(f'mlp_fc1: {lora_mlp_fc1}')
    print(f'mlp_fc2: {lora_mlp_fc2}')
    print(f'encoder: {lora_encoder}')
    print(f'decoder: {lora_decoder}')

    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc1.parameters():
                        param.requires_grad = True
                    # weigths.append(swin_block.mlp.fc1.weight)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc2.parameters():
                        param.requires_grad = True

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc1.parameters():
                        param.requires_grad = True
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    for param in swin_block.mlp.fc2.parameters():
                        param.requires_grad = True



    return model



def change_model_alpha(model, new_alpha, config):
    lora_mlp_fc1 = config['mlp_fc1'] # True
    lora_mlp_fc2 = config['mlp_fc2'] # True

    lora_encoder = config['encoder'] # [True,True,True,True]
    lora_decoder = config['decoder'] #[True,True,True,True]

    adapt_hyp = config['adapt_hyp'] # True

    
    # print(f'model beta: {new_alpha}')

    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    # weigths.append(swin_block.mlp.fc1.weight)
                    swin_block.mlp.fc1.lora.alpha = new_alpha
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2.lora.alpha = new_alpha

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc1.lora.alpha = new_alpha
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    swin_block.mlp.fc2.lora.alpha = new_alpha


    if adapt_hyp:
        for i,layer in enumerate(model.h_a):
            if isinstance(layer, ConvWithAdapter):
                model.h_a[i].adapter.alpha = new_alpha

        for i,layer in enumerate(model.h_mean_s):
            if isinstance(layer, ConvWithAdapter):
                model.h_mean_s[i].adapter.alpha = new_alpha

            elif isinstance(layer, SubpelConvWithAdapter):
                model.h_mean_s[i].adapter.alpha = new_alpha
            

        for i,layer in enumerate(model.h_scale_s):
            if isinstance(layer, ConvWithAdapter):
                model.h_scale_s[i].adapter.alpha = new_alpha
            elif isinstance(layer, SubpelConvWithAdapter):
                model.h_scale_s[i].adapter.alpha = new_alpha

