from matplotlib.pylab import beta

from utils.dataset import TestKodakDataset
import argparse
from utils import seed_all, compute_metrics, compute_lpips
from torch.utils.data import DataLoader

from lora import change_model_alpha_beta, get_lora_model, get_lora_lora_model



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

from .eval_continuous import inference, rec_dd

if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(description= "path to read the configuration of the evaluation")
    
    my_parser.add_argument("--test-dir", type = str, help = "Kodak Test directory", default = "/scratch/dataset/kodak/")
    my_parser.add_argument("--save-path", default='./res', type=str)

    my_parser.add_argument("--model", default='stf', type=str)
    my_parser.add_argument("--ckpt", default='../checkpoints/results/stf/rdp/_checkpoint.pth.tar', type=str)
    my_parser.add_argument("--label", default='beta-CALICE (STF)', type=str)
    my_parser.add_argument("--adapter-config", default='../configs/stf_8_8_all.yaml', type=str)
    my_parser.add_argument("--adapter-adapter-config", default='../configs/adapt_stf_8_8_all.yaml', type=str)
    


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


    total_res = rec_dd()

        
    ckpt = args.ckpt
    label = args.label
    adapt_config = args.adapter_config
    adapt_adapter_config = args.adapter_adapter_config
    model_type = args.model
    
    assert model_type in ['stf'], f'model {model_type} not yet implemented'

    net = models[model_type]()
    net, adapter_config = get_lora_model(net, adapt_config)
    net, adapter_adapter_config = get_lora_lora_model(net, adapt_adapter_config)
    
    net = net.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    net.load_state_dict(checkpoint["state_dict"])

    print(f"Testing model {label}")
    net.update(force = True)
    net.eval()
    alpha_rates = [ 1,0.9,0.8,0.6,0.45,0.35,0.25,0.15,0.05,0]
    alpha_perceptions = [ 1, 0 ]

    for i,alpha_per in enumerate(alpha_perceptions):
        
        bpp_list = []
        psnr_list = []
        mssim_list = []
        lpips_list = []
        
        print(f'alpha perception: {i+1}/{len(alpha_perceptions)}')
        print('-'*50)
        
        for j,alpha_rate in enumerate(alpha_rates):
            print(f'alpha rate: {j+1}/{len(alpha_rates)}')
            
            
            
            alpha_real = alpha_rate * adapter_config['alpha']
            beta_real = alpha_per * adapter_adapter_config['beta']
            
            change_model_alpha_beta(net, new_alpha=alpha_real, new_beta=beta_real)

            psnr_meter = AverageMeter()
            ms_ssim_meter = AverageMeter()
            lpips_meter = AverageMeter()

            bpps_meter = AverageMeter()

            for k,x in enumerate(tqdm(test_dataloader)):

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
            
        total_res[label][f'perception_{alpha_per}'] = {}
        total_res[label][f'perception_{alpha_per}']['psnr'] = psnr_list
        total_res[label][f'perception_{alpha_per}']['lpips'] = lpips_list
        total_res[label][f'perception_{alpha_per}']['bpp'] = bpp_list
        total_res[label][f'perception_{alpha_per}']['mssim'] = mssim_list
     

    print(total_res)
    file_name = f'{args.save_path}'
    with open(f'{file_name}.json', 'w') as outfile:
        json.dump(total_res, outfile, indent = 4)



