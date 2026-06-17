from utils.dataset import TestKodakDataset
import argparse
from utils import seed_all, compute_metrics, compute_lpips
from torch.utils.data import DataLoader
from compressai.zoo import cheng2020_attn
from conv_adapter import get_conv_adapt_model, change_model_alpha as chenge_model_conv_alpha
from lora import get_lora_model, change_model_alpha as chenge_model_lora_alpha
from mixed_adapter import get_mixed_adapt_model, change_model_alpha as chenge_model_mixed_alpha

from utils.engine import pad, crop  


import torch
from utils import compress_one_epoch, AverageMeter
from utils import plot_rate_distorsion
from custom_comp.zoo import models

from torchvision import transforms
from collections import defaultdict
import compressai
from tqdm import tqdm
import json





def rec_dd():
    return defaultdict(rec_dd)



@torch.no_grad()
def inference(model,x, x_padded, padding, eval_lpips = True):

    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])

    out_dec["x_hat"] = crop(out_dec["x_hat"], padding)
    # out_dec["x_hat"] = F.pad(out_dec["x_hat"], padding) 

    metrics = compute_metrics(x, out_dec["x_hat"], 255)

    lpips_util = compute_lpips()

    if eval_lpips:
        metrics_lpips = lpips_util.get_lpips(x, out_dec["x_hat"], 255)
        metrics['lpips'] = metrics_lpips

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels

    rate = bpp*num_pixels 

    return metrics, torch.tensor([bpp]), rate, out_dec["x_hat"]


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    
    my_parser.add_argument("--test-dir", type = str, help = "Kodak Test directory", default = "/scratch/dataset/kodak/")
    my_parser.add_argument("--save-path", default='./res', type=str)

    my_parser.add_argument("--model", default='stf', type=str)
    my_parser.add_argument("--ckpt", default='../checkpoints/results/stf/mse/_checkpoint_best.pth.tar', type=str)
    my_parser.add_argument("--label", default='CALICE (STF)', type=str)
    my_parser.add_argument("--adapter-config", default='../configs/stf_8_8_all.yml', type=str)


    my_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )


    args = my_parser.parse_args()

    seed_all(42)
    compressai.available_entropy_coders()[0]
    
    

    test_dataset = TestKodakDataset(data_dir=args.test_dir)

    test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=4)

    device = 'cuda'

    psnr_res = {}
    mssim_res = {}
    bpp_res = {}
    alpha_res = {}


    total_res = rec_dd()

        
    ckpt = args.ckpt
    label = args.label
    adapt_config = args.adapter_config
    model_type = args.model

    if model_type in ['stf','tcm']:
        net = models[model_type]()
        if model_type == 'stf':
            net, adapter_config = get_lora_model(net, adapt_config)
            change_model_alpha = chenge_model_lora_alpha
        else:
            net, adapter_config = get_mixed_adapt_model(net, adapt_config)
            change_model_alpha = chenge_model_mixed_alpha
    elif model_type == 'cheng-attn':
        net = cheng2020_attn(quality = 6)
        net, adapter_config = get_conv_adapt_model(net, adapt_config)
        change_model_alpha = chenge_model_conv_alpha
    else:
        raise NotImplementedError(f'model {model_type} not yet implemented')

    net = net.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    bpp_list = []
    psnr_list = []
    mssim_list = []
    lpips_list = []
    alpha_list = []

    print(f"Testing model {label}")
    net.update(force = True)
    net.eval()
    list_evaluation = [ 1,0.9,0.8,0.6,0.45,0.35,0.25,0.15,0.05,0]

    for i,alpha in enumerate(list_evaluation):

        print(f'{i+1}/{len(list_evaluation)}')
            
        alpha_real = alpha*adapter_config['alpha']
        change_model_alpha(net, new_alpha=alpha_real, config=adapter_config)

        psnr_meter = AverageMeter()
        ms_ssim_meter = AverageMeter()
        lpips_meter = AverageMeter()

        bpps_meter = AverageMeter()

        for j,x in enumerate(tqdm(test_dataloader)):

            x = x.to(device)
                
            x_padded, padding = pad(x, 128)

            metrics, bpp, _, _ = inference(net, x, x_padded, padding, eval_lpips=True)

            psnr_meter.update(metrics["psnr"])
            ms_ssim_meter.update(metrics["ms-ssim"])
            lpips_meter.update(metrics["lpips"])
            bpps_meter.update(bpp.item())

        bpp_list.append(bpps_meter.avg)
        psnr_list.append(psnr_meter.avg)
        lpips_list.append(lpips_meter.avg)
        mssim_list.append(ms_ssim_meter.avg)
        alpha_list.append(alpha)


    bpp_res[label] = bpp_list
    psnr_res[label] = psnr_list
    mssim_res[label] = mssim_list

    total_res[label]['psnr'] = psnr_list
    total_res[label]['lpips'] = lpips_list
    total_res[label]['bpp'] = bpp_list
    total_res[label]['mssim'] = mssim_list
    total_res[label]['alpha'] = alpha_list


    print(total_res)
    file_name = f'{args.save_path}'
    with open(f'{file_name}.json', 'w') as outfile:
        json.dump(total_res, outfile, indent = 4)



