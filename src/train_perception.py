import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from custom_comp.zoo import models
from compressai.zoo import cheng2020_attn, mbt2018_mean, bmshj2018_hyperprior, cheng2020_anchor

from opt import parse_args

from utils import AverageMeter, CustomDataParallel, RateDistortionPerceptionLoss, \
    configure_optimizers, save_checkpoint, seed_all, \
        get_dataloaders, lambda_interpolated, beta_interpolated

from utils.engine_rdp import train_one_epoch, test_epoch, compress_one_epoch
import os
import wandb
from lora import get_lora_model, get_lora_lora_model, change_model_alpha_beta

from utils import plot_rate_distortion_perception

import json

from pytorch_msssim import ms_ssim


def main():
    log_wandb = True
    # torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    print(args)


    if args.seed is not None:
        seed_all(args.seed)
        args.save_dir = f'{args.save_dir}_seed_{args.seed}'
    

    if log_wandb:
        wandb.init(
            project='CALICE',
            name=f'{args.save_dir}',
            config=vars(args)
        )

    print(f'\n\n--------------\nBeta list: {args.beta_perc}\n-----------------\n\n')
    print(f'\n\n--------------\nLambda list: {args.lmbda}\n-----------------\n\n')

    print(f'\n\nOptimizing for: {args.loss_type} !!!\n\n')
    assert args.loss_type == 'rdp'

    if args.save:
        print(f'Results will be saved in: {args.save_dir}')
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'n gpus: {torch.cuda.device_count()}')


    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        class_type=args.class_type,
        dataset_path=args.dataset,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        val_batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        device=device,
        test_dir=args.test_dir
    )

    if not args.compressai_checkpoint:
        net = models[args.model]()
    else:
        if args.model == 'cheng-attn':
            net = cheng2020_attn(quality=6, pretrained=True)
        # elif args.model == 'cheng-anchor':
        #     net = cheng2020_anchor(quality=6, pretrained=True)
        else:
            raise NotImplementedError(f'model: {args.model} not yet implemented')



    net = net.to(device)
    if args.loss_type == 'rdp':
        criterion = RateDistortionPerceptionLoss(
            lmbda_rate=None,
            lmbda_distortion=150,
            lmbda_lpips=2/args.beta_max, 
            beta=None
        )


    net, adapter_config = get_lora_model(net, args.adapter_config)

    if args.checkpoint is not None and not args.resume_train:  
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

    

    for param in net.parameters():
        param.requires_grad = False

    net, adapter_adapter_config = get_lora_lora_model(net, args.adapter_adapter_config)

    net = net.to(device)


    print('Trainable parameters')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name)
        
    total_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Total number of trainable parameters: {total_params}')


    optimizer, aux_optimizer = configure_optimizers(net, args, assert_intersection=False, opt = args.adapter_opt)
    if args.adapter_sched == 'lr_plateau':
        print('Using ReduceLROnPlateau scheduler')
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
    elif args.adapter_sched == 'cosine':
        print('Using Cosine scheduler')
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)



    last_epoch = 0
    best_val_loss = float("inf")

    if args.resume_train:
        assert args.checkpoint is not None
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        print('Loading elements from checkpoint to resume the training')
        last_epoch = checkpoint["epoch"] + 1

        optimizer.load_state_dict(checkpoint["optimizer"])
        if aux_optimizer is not None and checkpoint["aux_optimizer"] is not None:
            aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        else:
            print('aux optimizer not loaded!')
            
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        best_val_loss = checkpoint["best_val_loss"]
        

    if args.cuda and torch.cuda.device_count() > 1:
        print(f'Training on: {torch.cuda.device_count()} GPUs')
        net = CustomDataParallel(net)
    else:
        print(f'Training on a single GPU')


    first_epoch = True
    final_epochs = False
    skip_train = False
    if (last_epoch == args.epochs):
        last_epoch -= 1
        print(f'redo last epoch from resume: last_epoch: {last_epoch} epochs: {args.epochs}')
        skip_train = True

    frequency_compress = 250
    if args.compress_often:
        frequency_compress = 100

    for epoch in range(last_epoch, args.epochs):
        if (args.epochs - epoch) < 3:
            final_epochs = True
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Training 
        if not skip_train:
            loss_tot_train, \
            bpp_train, \
            mse_train, mse_scaled_train, \
            perception_train, perception_scaled_train, \
            distortion_train, aux_train_loss = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                alpha_perc = args.alpha_perc,
                beta_perc = args.beta_perc,
                lmbda_list = args.lmbda,
                beta_max = args.beta_max,
                linear_alpha = args.linear_alpha,
                adapter_config = adapter_config,
                adapter_adapter_config = adapter_adapter_config,
                change_model_alpha_beta = change_model_alpha_beta
            )
            if log_wandb:
                train_results = {
                        "train/epoch":epoch,
                        "train/loss": loss_tot_train,
                        "train/bpp_loss": bpp_train,
                        "train/aux_loss":aux_train_loss,
                        "train/leaning_rate": optimizer.param_groups[0]['lr'],
                        "train/mse_loss": mse_train,
                        "train/lpips_loss": perception_train,
                        "train/distortion_loss": distortion_train,
                        "train/mse_scaled_loss": mse_scaled_train,
                        "train/lpips_scaled_train": perception_scaled_train,

                    }
                # print(f'\nlogging on {epoch} (after train)\n')
                wandb.log(train_results, step = epoch)      


            # test on validation set
            lss = AverageMeter()

            if(epoch % 50 == 0 or first_epoch):
                
                alpha_eval = [1.0,0.8,0.6,0.4,0.2,0.0]
                    
                beta_eval = args.beta_perc 


                for beta in beta_eval:
                    lss_per_beta = AverageMeter()
                    for alpha in alpha_eval:
    
                        lmbda_loss = lambda_interpolated(alpha, lambda_max=args.lmbda[1], lambda_min=args.lmbda[0], linear = args.linear_alpha, inverse=True)
                        alpha_real = alpha * adapter_config['alpha']

                        beta_real = beta * adapter_adapter_config['beta']
                        beta_loss = beta_interpolated(beta, beta_max = args.beta_max)

                        
                        change_model_alpha_beta(net, new_alpha=alpha_real, new_beta=beta_real) #, config=adapter_adapter_config)
                        
                        loss_tot_val, \
                        bpp_loss_val, \
                        mse_loss_val, mse_loss_scaled_val, \
                        perception_loss_val, perception_loss_scaled_val, \
                        distortion_loss_val, \
                        aux_loss_val, \
                        psnr_val, ssim_val = test_epoch(
                                                epoch, 
                                                val_dataloader, 
                                                net, 
                                                criterion, 
                                                lmbda=lmbda_loss, 
                                                beta=beta_loss, 
                                                tag = 'Val')
        
                        # test on test
                        if log_wandb:

                            val_results = {
                                f"val_beta_{beta}/epoch_alpha_{alpha}":epoch,
                                f"val_beta_{beta}/loss_alpha_{alpha}": loss_tot_val,
                                f"val_beta_{beta}/bpp_loss_alpha_{alpha}": bpp_loss_val,
                                f"val_beta_{beta}/aux_loss_alpha_{alpha}":aux_loss_val,
                                f"val_beta_{beta}/psnr_alpha_{alpha}":psnr_val,
                                f"val_beta_{beta}/mssim_alpha_{alpha}":ssim_val,
                                f"val_beta_{beta}/mse_loss_alpha_{alpha}": mse_loss_val,
                                f"val_beta_{beta}/lpips_loss_alpha_{alpha}": perception_loss_val,
                                f"val_beta_{beta}/distortion_loss_alpha_{alpha}": distortion_loss_val,
                                f"val_beta_{beta}/mse_scaled_loss_alpha_{alpha}": mse_loss_scaled_val,
                                f"val_beta_{beta}/lpips_scaled_loss_alpha_{alpha}": perception_loss_scaled_val

                            }

                            wandb.log(val_results, step = epoch)   

                        lss.update(loss_tot_val)
                        lss_per_beta.update(loss_tot_val)

                    if log_wandb:
                        wandb.log({
                            f"val_avg_beta_{beta}/epoch": epoch,
                            f"val_avg_beta_{beta}/loss" : lss_per_beta.avg}, 
                        step = epoch)   
            

                if log_wandb:
                    # print(f'\nlogging on {epoch} (after all val avg)\n')
                    wandb.log({
                        "val_avg/epoch": epoch,
                        "val_avg/loss" : lss.avg}, 
                    step = epoch)   
            
            if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(lss.avg)
            else:
                lr_scheduler.step()

            # save checkpoint according to val_loss
            is_best_val = lss.avg < best_val_loss
            best_val_loss = min(lss.avg, best_val_loss)
            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "best_val_loss": best_val_loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict() if aux_optimizer is not None else None,
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best_val,
                    args.save_dir,
                    filename=f"_checkpoint.pth.tar"
                )


        # try to estimate metrics with the real Arithmetic coding (AC) 
        if (epoch%frequency_compress == 0 or first_epoch or final_epochs): # or is_best_val:

            print("Make actual compression")
            net.update(force = True)
            alpha_eval = [1.0,0.8,0.6,0.4,0.2,0.0]

            beta_eval = args.beta_perc 

            beta_eval.append(0.0)

            results = {}
            results['ours'] = {}

            for beta in beta_eval:
                results['ours'][f'beta_{beta}'] = {}
                bpp_list = []
                psnr_list = []
                mssim_list = []
                lpips_list = []
                for alpha in alpha_eval:
            
                    alpha_real = alpha * adapter_config['alpha']
                    beta_real = beta * adapter_adapter_config['beta']

                    change_model_alpha_beta(net, new_alpha=alpha_real, new_beta=beta_real)

                    bpp_ac, psnr_ac, mssim_ac, lpips_ac = compress_one_epoch(net, test_dataloader, device)
                    bpp_list.append(bpp_ac)
                    psnr_list.append(psnr_ac)
                    mssim_list.append(mssim_ac)
                    lpips_list.append(lpips_ac)
                    

                results['ours'][f'beta_{beta}']['bpp'] = bpp_list
                results['ours'][f'beta_{beta}']['PSNR'] = psnr_list
                results['ours'][f'beta_{beta}']['MS-SIM'] = mssim_list
                results['ours'][f'beta_{beta}']['LPIPS'] = lpips_list

                print("************************* our results **************************** ")
                print(f'Beta: {beta}')
                print(bpp_list)
                print("--")
                print(psnr_list)
                print("--")
                print(mssim_list)
                print("--")
                print(lpips_list)
                print("************************* end our result *************************************")


            plotting_res = {}

            plotting_res['STF-ours'] = results['ours']
            plot_rate_distortion_perception(plotting_res, epoch = epoch, eest='rate-distortion-perception', log_wandb = log_wandb, save_fig=False)

            filtered_results = {}

            for category, data in results.items():
                betas = data.keys()
                for beta in betas:
                    if beta not in ['beta_0.0','beta_1.0']:
                        continue
                    
                    filtered_results[beta] = results[category][beta]

            plotting_res['STF-ours'] = filtered_results
            plot_rate_distortion_perception(plotting_res, epoch = epoch, eest='rate-distortion-perception-extreme', log_wandb = log_wandb, save_fig=False)


        first_epoch = False
    
    if log_wandb:
        wandb.run.finish()


if __name__ == "__main__":
    main()