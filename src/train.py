import random
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from custom_comp.zoo import models
from compressai.zoo import cheng2020_attn, mbt2018_mean, bmshj2018_hyperprior, cheng2020_anchor

from opt import parse_args

from utils import AverageMeter, RateDistortionLoss, RateDistortionLoss_withLPIPS, CustomDataParallel, \
    train_one_epoch, test_epoch, compress_one_epoch, \
    configure_optimizers, save_checkpoint, seed_all, \
        get_dataloaders, lambda_interpolated
import os
import wandb
from lora import get_lora_model, get_merged_lora, change_model_alpha as chenge_model_lora_alpha
from conv_adapter import get_conv_adapt_model, change_model_alpha as chenge_model_conv_alpha

from mixed_adapter import get_mixed_adapt_model, change_model_alpha as chenge_model_mixed_alpha

from utils import plot_rate_distorsion

import json

from pytorch_msssim import ms_ssim

# here we train a set of adapter to improve the performance of the anchor model

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

    print(f'\n\n--------------\nAlpha list: {args.alpha_perc}\n-----------------\n\n')
    print(f'\n\n--------------\nLambda list: {args.lmbda}\n-----------------\n\n')

    print(f'\n\nOptimizing for: {args.loss_type} !!!\n\n')

    if args.save:
        print(f'Results will be saved in: {args.save_dir}')
        os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f'n gpus: {torch.cuda.device_count()}')



    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        class_type='natural',
        dataset_path=args.dataset,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        test_dir=args.test_dir
    )

    if not args.compressai_checkpoint:
        net = models[args.model]()
    else:
        if args.model == 'cheng-attn':
            net = cheng2020_attn(quality=args.compressai_quality, pretrained=True)
        # elif args.model == 'cheng-anchor':
        #     net = cheng2020_anchor(quality=args.compressai_quality, pretrained=True)
        else:
            raise NotImplementedError(f'model: {args.model} not yet implemented')


    net = net.to(device)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)

    if args.loss_type != 'lpips':
        criterion = RateDistortionLoss(lmbda=None, metric=args.loss_type)
        if args.fixed_lmbda is not None:
            print(f'Training with fixed lmbda value: {args.fixed_lmbda}')
            criterion.lmbda = args.fixed_lmbda
    else:
        criterion = RateDistortionLoss_withLPIPS(lmbda_rate=None, lmbda_distortion = 150, lmbda_perception = args.lmbda_percpetion, metric="mse")
        if args.fixed_lmbda is not None:
            print(f'Training with fixed lmbda value: {args.fixed_lmbda}')
            criterion.lmbda_rate = args.fixed_lmbda

    last_epoch = 0

    best_val_loss = float("inf")

    if args.checkpoint is not None and not args.resume_train:  
        print("Loading", args.checkpoint)

        if args.adapted_checkpoint:
            net, adapter_config = get_lora_model(net, args.adapter_checkpoint_config)

        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])

        net = net.to(device)

        if args.adapted_checkpoint:
            assert args.model == 'stf', f'Adapted checkpoint is only for STF model, {args.model} model not implemented yet'
            print('merging lora')
            net = get_merged_lora(net)

    adapter_config = None
    change_model_alpha = None
    if args.lora or args.conv_adapt or args.mixed_adapt:
        for param in net.parameters():
            param.requires_grad = False

        if args.fixed_lmbda is None:
            if args.lora:
                net, adapter_config = get_lora_model(net, args.adapter_config)
                change_model_alpha = chenge_model_lora_alpha
            elif args.conv_adapt:
                net, adapter_config = get_conv_adapt_model(net, args.adapter_config)
                change_model_alpha = chenge_model_conv_alpha
            elif args.mixed_adapt:
                net, adapter_config = get_mixed_adapt_model(net, args.adapter_config)
                change_model_alpha = chenge_model_mixed_alpha
        else:
            change_model_alpha = None
            args.alpha_perc = None

            if args.lora:
                net, adapter_config = get_lora_model(net, args.adapter_config)
            elif args.conv_adapt:
                net, adapter_config = get_conv_adapt_model(net, args.adapter_config)
            elif args.mixed_adapt:
                net, adapter_config = get_mixed_adapt_model(net, args.adapter_config)

        net = net.to(device)

        print('Trainable parameters')
        for name, param in net.named_parameters():
            if param.requires_grad:
                print(name)
        
        total_params  = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total number of trainable parameters: {total_params}')

        # redefine optimizers and scheduler
        
        optimizer, aux_optimizer = configure_optimizers(net, args, assert_intersection=False, opt = args.adapter_opt)
        if args.adapter_sched == 'lr_plateau':
            print('Using ReduceLROnPlateau scheduler')
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=4)
        elif args.adapter_sched == 'cosine':
            print('Using Cosine scheduler')
            lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


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
        frequency_compress = 50

    for epoch in range(last_epoch, args.epochs):
        if (args.epochs - epoch) < 3:
            final_epochs = True
            
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        # Training 
        if not skip_train:
            loss_tot_train, bpp_train, distortion_train, perception_train, aux_train_loss = train_one_epoch(
                net,
                criterion,
                train_dataloader,
                optimizer,
                aux_optimizer,
                epoch,
                args.clip_max_norm,
                alpha_perc = args.alpha_perc,
                lmbda_list = args.lmbda,
                linear_alpha = args.linear_alpha,
                inverse_lmbda = args.inverse_lmbda,
                config = adapter_config,
                change_model_alpha = change_model_alpha
            )
            if log_wandb:
                train_results = {
                        "train/epoch":epoch,
                        "train/loss": loss_tot_train,
                        "train/bpp_loss": bpp_train,
                        "train/aux_loss":aux_train_loss,
                        "train/leaning_rate": optimizer.param_groups[0]['lr']
                    }
                if type(criterion) == RateDistortionLoss:
                    if criterion.metric == ms_ssim:
                        train_results["train/ms_ssim_loss"] = distortion_train
                    else:
                        train_results["train/mse_loss"] = distortion_train
                else: 
                    train_results["train/mse_loss"] = distortion_train
                    train_results["train/lpips_loss"] = perception_train

                # print(f'\nlogging on {epoch} (after train)\n')
                wandb.log(train_results, step = epoch)      


            # test on validation set
            lss = AverageMeter()
            if args.alpha_perc is not None:
                alpha_eval = [0,0.2,0.4,0.6,0.8,1.0]
                iterations = len(alpha_eval)
            else:
                iterations = 1

            for i in range(iterations):
                if args.fixed_lmbda is None:
                    alpha = alpha_eval[i] #args.alpha_perc[i]

                    alpha_real = alpha * adapter_config['alpha']
                    change_model_alpha(net, new_alpha=alpha_real, config=adapter_config)
                    
                    lmbda = lambda_interpolated(alpha, lambda_max=args.lmbda[1], lambda_min=args.lmbda[0], linear = args.linear_alpha, inverse = args.inverse_lmbda)
                    print(f'alpha {alpha} -> lambda {lmbda}')
                
                else:
                    lmbda = None # I don't have to change lmbda


                loss_tot_val, bpp_loss_val, distortion_loss_val, perception_loss_val, aux_loss_val, psnr_val, ssim_val = \
                    test_epoch(epoch, val_dataloader, net, criterion, lmbda=lmbda, tag = 'Val')
                
                # test on test
                if log_wandb:

                    val_results = {
                        f"val/epoch_{i}":epoch,
                        f"val/loss_{i}": loss_tot_val,
                        f"val/bpp_loss_{i}": bpp_loss_val,
                        f"val/aux_loss_{i}":aux_loss_val,
                        f"val/psnr_{i}":psnr_val,
                        f"val/mssim_{i}":ssim_val
                    }

                    if type(criterion) == RateDistortionLoss:
                        if criterion.metric == ms_ssim:
                            val_results[f"val/ms_ssim_loss_{i}"] = distortion_loss_val
                        else:
                            val_results[f"val/mse_loss_{i}"] = distortion_loss_val
                    else:
                        val_results[f"val/mse_loss_{i}"] = distortion_loss_val
                        val_results[f"val/lpips_loss_{i}"] = perception_loss_val

                    # print(f'\nlogging on {epoch} (after val)\n')
                    wandb.log(val_results, step = epoch)   

                lss.update(loss_tot_val)
                # test_loss.update(loss_tot_test)

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
            bpp_list = []
            psnr_list = []
            mssim_list = []
            lpips_list = []

            print("Make actual compression")
            net.update(force = True)

            if args.fixed_lmbda is None:
                list_evaluation = [ 1,0.9,0.8,0.6,0.45,0.35,0.25,0.15,0.05,0]
            else:
                list_evaluation = [ 1 ]

            for alpha in list_evaluation:
                
                if change_model_alpha is not None:
                    alpha_real = alpha*adapter_config['alpha']
                    change_model_alpha(net, new_alpha=alpha_real, config=adapter_config)

                bpp_ac, psnr_ac, mssim_ac, lpips_ac = compress_one_epoch(net, test_dataloader, device)
                bpp_list.append(bpp_ac)
                psnr_list.append(psnr_ac)
                mssim_list.append(mssim_ac)
                lpips_list.append(lpips_ac)

                if log_wandb:
                    wandb.log({
                        f"test_compress/bpp_{alpha}": bpp_ac,
                        f"test_compress/psnr_{alpha}": psnr_ac,
                        f"test_compress/mssim_{alpha}": mssim_ac,
                        f"test_compress/lpips_{alpha}": lpips_ac
                    },step = epoch)  
            psnr_res = {}
            mssim_res = {}
            lpips_res = {}
            bpp_res = {}


            bpp_res["ours"] = bpp_list
            psnr_res["ours"] = psnr_list
            mssim_res["ours"] = mssim_list
            lpips_res["ours"] = lpips_list


            print("************************* our results **************************** ")
            print('bpp:')
            print(bpp_list)
            print("--")
            print('psnr:')
            print(psnr_list)
            print("--")
            print('mssim:')
            print(mssim_list)
            print("--")
            print('lpips:')
            print(lpips_list)
            print("************************* end our result *************************************")


            plot_rate_distorsion(bpp_res, psnr_res, epoch, eest="compression_psnr", metric = 'PSNR',save_fig=False, log_wandb=log_wandb)
            plot_rate_distorsion(bpp_res, mssim_res, epoch, eest="compression_mssim", metric = 'MS-SSIM',save_fig=False, log_wandb=log_wandb, is_psnr=False)
            plot_rate_distorsion(bpp_res, lpips_res, epoch, eest="compression_lpips", metric = 'LPIPS',save_fig=False, log_wandb=log_wandb, is_psnr=False)
        
        first_epoch = False
    
    if log_wandb:
        wandb.run.finish()


if __name__ == "__main__":
    main()
