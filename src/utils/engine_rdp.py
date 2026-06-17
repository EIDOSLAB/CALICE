import torch
from utils.functions import compute_psnr, compute_metrics, compute_msssim, compute_lpips

import torch.nn.functional as F

import random 
import numpy as np
from pytorch_msssim import ms_ssim
from .loss import RateDistortionPerceptionLoss
from .engine import AverageMeter, lambda_interpolated


def beta_interpolated(beta, beta_max = 5.12):
    return beta*beta_max

def train_one_epoch(
        model, 
        criterion:RateDistortionPerceptionLoss, 
        train_dataloader, 
        optimizer, 
        aux_optimizer, 
        epoch, 
        clip_max_norm,
        alpha_perc = None,
        beta_perc = None,
        lmbda_list = None,
        beta_max = 5.12,
        linear_alpha = True,
        adapter_config = None,
        adapter_adapter_config = None,
        change_model_alpha_beta = None):
    
    model.train()
    device = next(model.parameters()).device


    loss_tot_metric = AverageMeter()
    bpp_loss_metric = AverageMeter()
    mse_loss_metric = AverageMeter()
    perception_loss_metric = AverageMeter()
    distortion_loss_metric = AverageMeter()
    aux_loss_metric = AverageMeter()

    mse_loss_scaled_metric = AverageMeter()
    perception_loss_scaled_metric = AverageMeter()

    for i, d in enumerate(train_dataloader):

        # if i>5:
        #     break

        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()


        index = random.randint(0,len(alpha_perc) - 1)
        alpha_picked = alpha_perc[index]
        
        criterion.lmbda_rate = lambda_interpolated(alpha_picked, lambda_max=lmbda_list[1], lambda_min=lmbda_list[0], linear=linear_alpha, inverse = True)
        
        alpha_real = alpha_picked*adapter_config['alpha']  
        
        index = random.randint(0,len(beta_perc) - 1)
        beta_picked = beta_perc[index]

        criterion.beta = beta_interpolated(beta_picked, beta_max = beta_max)

        beta_real = beta_picked*adapter_adapter_config['beta']
        
        if change_model_alpha_beta is not None:

            change_model_alpha_beta(model, new_alpha = alpha_real, new_beta = beta_real)
        
        
        out_net = model(d)

        if torch.isnan(out_net["x_hat"]).any() or torch.isinf(out_net["x_hat"]).any():
            print("\n\nWARNING: x_hat is NaN or Inf!\n")

        out_criterion = criterion(out_net, d)

        if torch.isnan(out_criterion["loss"]).any() or torch.isinf(out_criterion["loss"]).any():
            print("\n\nWARNING: Loss is NaN or Inf! Skipping this batch.\n")
            continue 
        
        out_criterion["loss"].backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        
        
        optimizer.step()

        aux_loss = model.aux_loss()
        if aux_optimizer is not None:
            
            aux_loss.backward()
            aux_optimizer.step()

        distortion_criterion = out_criterion["distortion"].item()
        
        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tDistortion loss: {distortion_criterion / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}" 
            )

        loss_tot_metric.update(out_criterion["loss"].clone().detach())
        bpp_loss_metric.update(out_criterion["bpp_loss"].clone().detach())
        
        mse_loss_metric.update(out_criterion["mse_loss"].clone().detach())
        perception_loss_metric.update(out_criterion["perceptual_loss"].clone().detach())
        distortion_loss_metric.update(out_criterion["distortion"].clone().detach())


        mse_loss_scaled_metric.update(out_criterion["mse_loss_scaled"].clone().detach())
        perception_loss_scaled_metric.update(out_criterion["perceptual_loss_scaled"].clone().detach())



        aux_loss_metric.update(aux_loss.clone().detach())

    return  loss_tot_metric.avg, \
            bpp_loss_metric.avg, \
            mse_loss_metric.avg, mse_loss_scaled_metric.avg, \
            perception_loss_metric.avg, perception_loss_scaled_metric.avg, \
            distortion_loss_metric.avg, \
            aux_loss_metric.avg


def test_epoch(epoch, test_dataloader, model, criterion:RateDistortionPerceptionLoss, lmbda = None, beta = None, tag = 'Val'):
    model.eval()
    device = next(model.parameters()).device

    loss_tot_metric = AverageMeter()
    bpp_loss_metric = AverageMeter()
    distortion_loss_metric = AverageMeter()
    perception_loss_metric = AverageMeter()
    mse_loss_metric = AverageMeter()
    aux_loss_metric = AverageMeter()

    psnr_metric = AverageMeter()
    ssim_metric = AverageMeter()


    mse_loss_scaled_metric = AverageMeter()
    perception_loss_scaled_metric = AverageMeter()
    
    criterion.lmbda_rate = lmbda
    criterion.beta = beta


    with torch.no_grad():
        for i,d in enumerate(test_dataloader):

            # if i > 5:
            #     break
            d = d.to(device)
            out_net = model(d)

            out_criterion = criterion(out_net, d)

            psnr_metric.update(compute_psnr(d, out_net["x_hat"]))
            ssim_metric.update(compute_msssim(d, out_net["x_hat"]))


            loss_tot_metric.update(out_criterion["loss"])
            bpp_loss_metric.update(out_criterion["bpp_loss"])
            distortion_loss_metric.update(out_criterion["distortion"])
            perception_loss_metric.update(out_criterion["perceptual_loss"])
            mse_loss_metric.update(out_criterion["mse_loss"])
            aux_loss_metric.update(model.aux_loss())


            mse_loss_scaled_metric.update(out_criterion["mse_loss_scaled"])
            perception_loss_scaled_metric.update(out_criterion["perceptual_loss_scaled"])


    print(
        f"{tag} epoch {epoch}: Average losses:"
        f"\tLoss: {loss_tot_metric.avg:.3f} |"
        f"\tDistortion loss: {distortion_loss_metric.avg / 3:.3f} |"
        f"\tBpp loss: {bpp_loss_metric.avg:.2f} |"
        f"\tAux loss: {aux_loss_metric.avg:.2f}"
    )
    return  loss_tot_metric.avg, \
            bpp_loss_metric.avg, \
            mse_loss_metric.avg, mse_loss_scaled_metric.avg, \
            perception_loss_metric.avg, perception_loss_scaled_metric.avg, \
            distortion_loss_metric.avg, \
            aux_loss_metric.avg, \
            psnr_metric.avg, ssim_metric.avg





def compress_one_epoch(model, test_dataloader, device):
    bpp_metric = AverageMeter()
    psnr_metric = AverageMeter()
    mssim_metric = AverageMeter()
    lpips_metric = AverageMeter()


    lpips_util = compute_lpips()
    with torch.no_grad():
        for i,d in enumerate(test_dataloader): 
            # if i > 5:
            #     break
            d = d.to(device)


            x_padded, padding = pad(d, 128)
            
            
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            
            # out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
            out_dec["x_hat"] = crop(out_dec["x_hat"], padding)

            
            metrics = compute_metrics(d, out_dec["x_hat"], 255)
            metrics_lpips = lpips_util.get_lpips(d, out_dec["x_hat"], 255)
            metrics['lpips'] = metrics_lpips

            num_pixels = d.size(0) * d.size(2) * d.size(3)
            bpp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_pixels
            
            psnr_metric.update(metrics["psnr"])
            mssim_metric.update(metrics["ms-ssim"]) 
            lpips_metric.update(metrics["lpips"])
            bpp_metric.update(bpp)  

    print(
        f"Average metrics:"
        f"\tPSNR: {psnr_metric.avg:.3f} |"
        f"\tMSSIM: {mssim_metric.avg:.3f} |"
        f"\tLPIPPS: {lpips_metric.avg:.3f} |"
        f"\tBpp: {bpp_metric.avg:.2f}"
    )

    
    return bpp_metric.avg, psnr_metric.avg, mssim_metric.avg, lpips_metric.avg


def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )