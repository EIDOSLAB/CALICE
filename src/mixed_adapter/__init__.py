from functools import partial
from conv_adapter.layers import ConvWithAdapter, SubpelConvWithAdapter
from lora.layers import LinearWithLoRA
from custom_comp.models.tcm import ConvTransBlock
import sys
import torch.nn as nn
import yaml

from custom_comp.zoo import models
import torch
from custom_comp.zoo.pretrained import load_pretrained

import math


# from custom_comp.layers import BasicLayer, SwinTransformerBlock
from compressai.zoo import cheng2020_attn, cheng2020_anchor
from compressai.layers import ResidualBlock, AttentionBlock, ResidualBlockWithStride, ResidualBlockUpsample


def swish(x, beta=1.0):
    return x * torch.sigmoid(x * beta)

def get_mixed_adapt_model(model, config, force_alpha = None):
    print('Get MixedAdapter model!')

    with open(config, 'r') as file:
        adapt_conf = yaml.safe_load(file)

    adapt_rank = adapt_conf['r'] # 8

    if force_alpha is None:
        adapt_alpha = adapt_conf['alpha'] # 16
    else:
        adapt_alpha = force_alpha


    # ConvTransBlock
    adapt_conv_trans = adapt_conf['adapt_conv_trans'] # True

    # Attention Block (ConvTransBlock)
    adapt_mlp_fc1_conv_trans = adapt_conf['adapt_mlp_fc1_conv_trans'] # True
    adapt_mlp_fc2_conv_trans = adapt_conf['adapt_mlp_fc2_conv_trans'] # True
    
    # ?? Block (ConvTransBlock)
    adapt_conv1_1_conv_trans = adapt_conf['adapt_conv1_1_conv_trans'] # True
    adapt_conv1_2_conv_trans = adapt_conf['adapt_conv1_2_conv_trans'] # True

    # Residual Block (ConvTransBlock)
    adapt_conv1_resblock_conv_trans = adapt_conf['adapt_conv1_resblock_conv_trans'] # True
    adapt_conv2_resblock_conv_trans = adapt_conf['adapt_conv2_resblock_conv_trans'] # True


    # downsample (ResidualBlockWithStride)
    adapt_residual_stride = adapt_conf['adapt_residual_stride'] # True
    adapt_conv1_residual_stride = adapt_conf['adapt_conv1_residual_stride'] # True
    adapt_conv2_residual_stride = adapt_conf['adapt_conv2_residual_stride'] # True

    # upsample (ResidualBlockUpsample)
    adapt_residual_upsample = adapt_conf['adapt_residual_upsample'] # True
    adapt_subpel_conv_residual_upsample = adapt_conf['adapt_subpel_conv_residual_upsample'] # True
    adapt_conv_residual_upsample = adapt_conf['adapt_conv_residual_upsample'] # True



    adapt_act = adapt_conf['activation']
    divide_rank = adapt_conf['divide_rank']


    print('Configs:')
    print(f'r: {adapt_rank}')
    print(f'alpha: {adapt_alpha}')
    print(f'activation: {adapt_act}')

    print(f'adapt ConvTransBlock: {adapt_conv_trans}')
    print(f'ConvTransBlock - fc1 (mlp): {adapt_mlp_fc1_conv_trans}')
    print(f'ConvTransBlock - fc2 (mlp): {adapt_mlp_fc2_conv_trans}')
    print(f'ConvTransBlock - conv_1_1: {adapt_conv1_1_conv_trans}')
    print(f'ConvTransBlock - conv_1_2: {adapt_conv1_2_conv_trans}')
    print(f'ConvTransBlock - conv1 (ResidualBlock): {adapt_conv1_resblock_conv_trans}')
    print(f'ConvTransBlock - conv2 (ResidualBlock): {adapt_conv2_resblock_conv_trans}')

    print(f'(Downsample) adapt ResidualBlockWithStride: {adapt_residual_stride}')
    print(f'(Downsample) ResidualBlockWithStride - conv1: {adapt_conv1_residual_stride}')
    print(f'(Downsample) ResidualBlockWithStride - conv2: {adapt_conv2_residual_stride}')
    
    print(f'(Upsample) adapt ResidualBlockUpsample: {adapt_residual_upsample}')
    print(f'(Upsample) ResidualBlockUpsample - subpel_conv: {adapt_subpel_conv_residual_upsample}')
    print(f'(Upsample) ResidualBlockUpsample - conv: {adapt_conv_residual_upsample}')


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
    assign_subpel_conv_adapter = partial(SubpelConvWithAdapter, rank=adapt_rank, alpha=adapt_alpha, activation = act, divide_rank = divide_rank)
    assign_lora = partial(LinearWithLoRA, rank=adapt_rank, alpha=adapt_alpha, activation = act, divide_rank = divide_rank)
    
    for _,layer in enumerate(model.g_a): # encoder

        if isinstance(layer, ResidualBlockWithStride) and adapt_residual_stride:
            if adapt_conv1_residual_stride:
                layer.conv1 = assign_conv_adapter(layer.conv1)
            if adapt_conv2_residual_stride:
                layer.conv2 = assign_conv_adapter(layer.conv2)

        elif isinstance(layer, ConvTransBlock) and adapt_conv_trans:
            if adapt_mlp_fc1_conv_trans:
                layer.trans_block.mlp[0] = assign_lora(layer.trans_block.mlp[0])
            if adapt_mlp_fc2_conv_trans:
                layer.trans_block.mlp[2] = assign_lora(layer.trans_block.mlp[2])

            if adapt_conv1_1_conv_trans:
                layer.conv1_1 = assign_conv_adapter(layer.conv1_1)
            if adapt_conv1_2_conv_trans:
                layer.conv1_2 = assign_conv_adapter(layer.conv1_2)

            if adapt_conv1_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv1 = assign_conv_adapter(layer.conv_block.conv1)
            if adapt_conv2_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv2 = assign_conv_adapter(layer.conv_block.conv2)


    for _,layer in enumerate(model.g_s): # decoder

        if isinstance(layer, ResidualBlockUpsample) and adapt_residual_upsample:
            if adapt_subpel_conv_residual_upsample:
                layer.subpel_conv = assign_subpel_conv_adapter(layer.subpel_conv)
            if adapt_conv_residual_upsample:
                layer.conv = assign_conv_adapter(layer.conv)

        elif isinstance(layer, ConvTransBlock) and adapt_conv_trans:
            if adapt_mlp_fc1_conv_trans:
                layer.trans_block.mlp[0] = assign_lora(layer.trans_block.mlp[0])
            if adapt_mlp_fc2_conv_trans:
                layer.trans_block.mlp[2] = assign_lora(layer.trans_block.mlp[2])

            if adapt_conv1_1_conv_trans:
                layer.conv1_1 = assign_conv_adapter(layer.conv1_1)
            if adapt_conv1_2_conv_trans:
                layer.conv1_2 = assign_conv_adapter(layer.conv1_2)

            if adapt_conv1_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv1 = assign_conv_adapter(layer.conv_block.conv1)
            if adapt_conv2_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv2 = assign_conv_adapter(layer.conv_block.conv2)

    return model,adapt_conf



def change_model_alpha(model, new_alpha, config):
    # print('Change alpha of Mixed Model')
    # ConvTransBlock
    adapt_conv_trans = config['adapt_conv_trans'] # True

    # Attention Block (ConvTransBlock)
    adapt_mlp_fc1_conv_trans = config['adapt_mlp_fc1_conv_trans'] # True
    adapt_mlp_fc2_conv_trans = config['adapt_mlp_fc2_conv_trans'] # True
    
    # ?? Block (ConvTransBlock)
    adapt_conv1_1_conv_trans = config['adapt_conv1_1_conv_trans'] # True
    adapt_conv1_2_conv_trans = config['adapt_conv1_2_conv_trans'] # True

    # Residual Block (ConvTransBlock)
    adapt_conv1_resblock_conv_trans = config['adapt_conv1_resblock_conv_trans'] # True
    adapt_conv2_resblock_conv_trans = config['adapt_conv2_resblock_conv_trans'] # True


    # downsample (ResidualBlockWithStride)
    adapt_residual_stride = config['adapt_residual_stride'] # True
    adapt_conv1_residual_stride = config['adapt_conv1_residual_stride'] # True
    adapt_conv2_residual_stride = config['adapt_conv2_residual_stride'] # True

    # upsample (ResidualBlockUpsample)
    adapt_residual_upsample = config['adapt_residual_upsample'] # True
    adapt_subpel_conv_residual_upsample = config['adapt_subpel_conv_residual_upsample'] # True
    adapt_conv_residual_upsample = config['adapt_conv_residual_upsample'] # True

    i = 0
    for _,layer in enumerate(model.g_a): # encoder

        if isinstance(layer, ResidualBlockWithStride) and adapt_residual_stride:
            if adapt_conv1_residual_stride:
                layer.conv1.adapter.alpha = new_alpha
            if adapt_conv2_residual_stride:
                layer.conv2.adapter.alpha = new_alpha

        
        elif isinstance(layer, ConvTransBlock) and adapt_conv_trans:
            if adapt_mlp_fc1_conv_trans:
                layer.trans_block.mlp[0].lora.alpha = new_alpha
            if adapt_mlp_fc2_conv_trans:
                layer.trans_block.mlp[2].lora.alpha = new_alpha

            if adapt_conv1_1_conv_trans:
                layer.conv1_1.adapter.alpha = new_alpha
            if adapt_conv1_2_conv_trans:
                layer.conv1_2.adapter.alpha = new_alpha

            if adapt_conv1_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv1.adapter.alpha = new_alpha
            if adapt_conv2_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv2.adapter.alpha = new_alpha



    for _,layer in enumerate(model.g_s): # decoder

        if isinstance(layer, ResidualBlockUpsample) and adapt_residual_upsample:
            if adapt_subpel_conv_residual_upsample:
                layer.subpel_conv.adapter.alpha = new_alpha
            if adapt_conv_residual_upsample:
                layer.conv.adapter.alpha = new_alpha


        elif isinstance(layer, ConvTransBlock) and adapt_conv_trans:
            if adapt_mlp_fc1_conv_trans:
                layer.trans_block.mlp[0].lora.alpha = new_alpha
            if adapt_mlp_fc2_conv_trans:
                layer.trans_block.mlp[2].lora.alpha = new_alpha

            if adapt_conv1_1_conv_trans:
                layer.conv1_1.adapter.alpha = new_alpha
            if adapt_conv1_2_conv_trans:
                layer.conv1_2.adapter.alpha = new_alpha

            if adapt_conv1_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv1.adapter.alpha = new_alpha
            if adapt_conv2_resblock_conv_trans:
                assert isinstance(layer.conv_block, ResidualBlock) 
                layer.conv_block.conv2.adapter.alpha = new_alpha




if __name__ == '__main__':

    def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255):
        return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

    def compute_bpp(out_net):
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in out_net['likelihoods'].values()).item()


    def compress_img(x, net, device = 'cuda'):

        def compute_metrics(org, rec, max_val = 255):
            metrics = {}
            org = (org * max_val).clamp(0, max_val).round()
            rec = (rec * max_val).clamp(0, max_val).round()
            metrics["psnr"] = psnr(org, rec).item()
            return metrics

        x = x.to(device)
        net = net.to(device)
        # with torch.no_grad():
        out_net = net.forward(x)

        metrics = compute_metrics(x, out_net["x_hat"], 255)
        bpp = compute_bpp(out_net)

        return metrics, bpp
    
    
    from custom_comp.zoo import models
    from torchvision import transforms
    from PIL import Image


    device = "cuda"


    # load image
    img = Image.open('kodim01.png').convert('RGB')
    x = transforms.ToTensor()(img).unsqueeze(0)

    model = models['tcm']()
    anchor = '../checkpoints/anchors/tcm_0.05.pth.tar'
    checkpoint = torch.load(anchor, map_location='cpu')

    model.load_state_dict(checkpoint["state_dict"])

    model = model.to(device)

    metrics, bpp = compress_img(x, model, device=device)

    print(f"PSNR: {metrics['psnr']:.2f}dB")
    print(f'Bit-rate: {bpp:.3f} bpp')
    print('-'*15)


    model, adapt_config = get_mixed_adapt_model(model, '../configs/tcm_8_1_all.yml')
 
    metrics, bpp = compress_img(x, model, device=device)

    print(f"PSNR: {metrics['psnr']:.2f}dB")
    print(f'Bit-rate: {bpp:.3f} bpp')
    print('-'*15)
    

    change_model_alpha(model, new_alpha=0.0, config = adapt_config)

    metrics, bpp = compress_img(x, model, device=device)

    print(f"PSNR: {metrics['psnr']:.2f}dB")
    print(f'Bit-rate: {bpp:.3f} bpp')
    print('-'*15)

    # print(model.g_s)