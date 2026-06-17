import torch
import torch.nn as nn
from functools import partial

from .layers import LoraWithLora, LinearWithLoRA, LoRALayer
from custom_comp.layers import BasicLayer, SwinTransformerBlock

import yaml

def swish(x, beta=1.0):
    return x * torch.sigmoid(x * beta)

def get_lora_lora_model(model, config, force_beta = None):
    print('Get LoRA LoRA model!')

    with open(config, 'r') as file:
        lora_conf = yaml.safe_load(file)

    lora_r = lora_conf['r'] # 8

    if force_beta is None:
        lora_beta = lora_conf['beta'] # 16
    else:
        lora_beta = force_beta

    lora_mlp_fc1 = lora_conf['mlp_fc1'] # True
    lora_mlp_fc2 = lora_conf['mlp_fc2'] # True

    lora_encoder = lora_conf['encoder'] # [True,True,True,True]
    lora_decoder = lora_conf['decoder'] #[True,True,True,True]

    lora_act = lora_conf['activation'] # [True,True,True,True]
    divide_rank = lora_conf['divide_rank'] # True 

    adapt_hyp = lora_conf['adapt_hyp'] # True 

    assert not adapt_hyp, f'Adapting hyp not yet implemented'


    print('Configs:')
    print(f'r: {lora_r}')
    print(f'beta: {lora_beta}')
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

    assign_lora = partial(LoraWithLora, rank=lora_r, beta=lora_beta, activation = act, divide_rank = divide_rank)

    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer) and lora_encoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    # weigths.append(swin_block.mlp.fc1.weight)
                    assert type(swin_block.mlp.fc1) == LinearWithLoRA
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    assert type(swin_block.mlp.fc2) == LinearWithLoRA
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer) and lora_decoder[i]:
            for swin_block in layer.blocks:
                # print(swin_block)
                if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                    assert type(swin_block.mlp.fc1) == LinearWithLoRA
                    swin_block.mlp.fc1 = assign_lora(swin_block.mlp.fc1)
                if lora_mlp_fc2 and isinstance(swin_block, SwinTransformerBlock):
                    assert type(swin_block.mlp.fc2) == LinearWithLoRA
                    swin_block.mlp.fc2 = assign_lora(swin_block.mlp.fc2)



    return model,lora_conf



def change_model_alpha_beta(model, new_alpha, new_beta): #, config_adapter, config_adapter_adapter):
    '''
    config_adapter:         used for changing alpha
    config_adapter_adapter: used for changing beta
    '''
    # alpha_lora_mlp_fc1 = config_adapter['mlp_fc1'] # True
    # alpha_lora_mlp_fc2 = config_adapter['mlp_fc2'] # True

    # beta_lora_mlp_fc1 = config_adapter_adapter['mlp_fc1'] # True
    # beta_lora_mlp_fc2 = config_adapter_adapter['mlp_fc2'] # True

    # alpha_lora_encoder = config_adapter['encoder'] # [True,True,True,True]
    # alpha_lora_decoder = config_adapter['decoder'] #[True,True,True,True]

    # beta_lora_encoder = config_adapter_adapter['encoder'] # [True,True,True,True]
    # beta_lora_decoder = config_adapter_adapter['decoder'] #[True,True,True,True]

    tot = 0
    for i,layer in enumerate(model.layers): # encoder
        if isinstance(layer, BasicLayer):
            for swin_block in layer.blocks:
                # if lora_mlp_fc1 and isinstance(swin_block, SwinTransformerBlock):
                if isinstance(swin_block.mlp.fc1, LoraWithLora):
                    assert type(swin_block.mlp.fc1.linear_with_lora) == LinearWithLoRA
                    assert type(swin_block.mlp.fc1.lora) == LoRALayer
                    # print('ENCODER (fc1): changing alpha and beta')
                    tot +=1
                    swin_block.mlp.fc1.linear_with_lora.lora.alpha = new_alpha
                    swin_block.mlp.fc1.lora.alpha = new_beta
                elif isinstance(swin_block.mlp.fc1, LinearWithLoRA):
                    assert type(swin_block.mlp.fc1.lora) == LoRALayer
                    # print('ENCODER (fc1): changing only alpha')
                    tot +=1
                    swin_block.mlp.fc1.lora.alpha = new_alpha


                if isinstance(swin_block.mlp.fc2, LoraWithLora):
                    assert type(swin_block.mlp.fc2.linear_with_lora) == LinearWithLoRA
                    assert type(swin_block.mlp.fc2.lora) == LoRALayer
                    # print('ENCODER (fc2): changing alpha and beta')
                    tot +=1
                    swin_block.mlp.fc2.linear_with_lora.lora.alpha = new_alpha
                    swin_block.mlp.fc2.lora.alpha = new_beta
                elif isinstance(swin_block.mlp.fc2, LinearWithLoRA):
                    assert type(swin_block.mlp.fc2.lora) == LoRALayer
                    # print('ENCODER (fc2): changing only alpha')
                    tot +=1
                    swin_block.mlp.fc2.lora.alpha = new_alpha

    for i,layer in enumerate(model.syn_layers): # decoder
        if isinstance(layer, BasicLayer):
            for swin_block in layer.blocks:
                if isinstance(swin_block.mlp.fc1, LoraWithLora):
                    assert type(swin_block.mlp.fc1.linear_with_lora) == LinearWithLoRA
                    assert type(swin_block.mlp.fc1.lora) == LoRALayer
                    # print('Decoder (fc1): changing alpha and beta')
                    tot +=1

                    swin_block.mlp.fc1.linear_with_lora.lora.alpha = new_alpha
                    swin_block.mlp.fc1.lora.alpha = new_beta
                elif isinstance(swin_block.mlp.fc1, LinearWithLoRA):
                    assert type(swin_block.mlp.fc1.lora) == LoRALayer
                    # print('Decoder (fc1): changing only alpha')
                    tot +=1

                    swin_block.mlp.fc1.lora.alpha = new_alpha


                if isinstance(swin_block.mlp.fc2, LoraWithLora):
                    assert type(swin_block.mlp.fc2.linear_with_lora) == LinearWithLoRA
                    assert type(swin_block.mlp.fc2.lora) == LoRALayer
                    # print('Decoder (fc2): changing alpha and beta')
                    tot +=1

                    swin_block.mlp.fc2.linear_with_lora.lora.alpha = new_alpha
                    swin_block.mlp.fc2.lora.alpha = new_beta
                elif isinstance(swin_block.mlp.fc2, LinearWithLoRA):
                    assert type(swin_block.mlp.fc2.lora) == LoRALayer
                    # print('Decoder (fc2): changing only alpha')
                    tot +=1

                    swin_block.mlp.fc2.lora.alpha = new_alpha

    # print(f'TOT: {tot}')