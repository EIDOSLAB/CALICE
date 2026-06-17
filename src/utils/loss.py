import torch.nn as nn
import torch
import math
from pytorch_msssim import ms_ssim

from compressai.registry import register_criterion
import lpips
import sys


@register_criterion("RateDistortionLoss")
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=0.01, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        elif metric == "ms-ssim":
            self.metric = ms_ssim
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        if self.metric == ms_ssim:
            out["ms_ssim_loss"] = self.metric(output["x_hat"], target, data_range=1)
            out["distortion"] = 1 - out["ms_ssim_loss"]
        else:
            out["mse_loss"] = self.metric(output["x_hat"], target)
            out["distortion"] = 255**2 * out["mse_loss"]

        out["loss"] = self.lmbda * out["distortion"] + out["bpp_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]
        


class RateDistortionLoss_withLPIPS(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda_rate=0.048, lmbda_distortion=150, lmbda_perception = 1, metric="mse", return_type="all"):
        super().__init__()
        if metric == "mse":
            self.metric = nn.MSELoss()
        else:
            raise NotImplementedError(f"{metric} is not implemented!")
        
        self.lpips_loss = lpips.LPIPS(net='alex').cuda()

        self.lmbda_rate = lmbda_rate
        self.return_type = return_type

        self.lmbda_distortion = lmbda_distortion
        self.lmbda_perception = lmbda_perception


    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        
        out["mse_loss"] = self.metric(output["x_hat"], target)
        out["distortion"] = self.lmbda_distortion * out["mse_loss"]
        
        out["perceptual_loss"] = self.lmbda_perception * torch.mean(self.lpips_loss(output["x_hat"], target))

        out["loss"] = self.lmbda_rate *out["bpp_loss"] + out["distortion"] + out["perceptual_loss"]
        if self.return_type == "all":
            return out
        else:
            return out[self.return_type]




class RateDistortionPerceptionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda_rate=0.048, lmbda_distortion=150, lmbda_lpips=0.1, beta = 1.):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lpips_loss = lpips.LPIPS(net='alex').cuda()
        self.lmbda_rate = lmbda_rate
        self.lmbda_distortion = lmbda_distortion
        self.lmbda_lpips = lmbda_lpips

        self.beta = beta

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["mse_loss"] = self.mse(output["x_hat"], target)
        out["mse_loss_scaled"] =  self.lmbda_distortion * out["mse_loss"]

        out['perceptual_loss'] = torch.mean(self.lpips_loss(output["x_hat"], target)) 
        out['perceptual_loss_scaled'] = self.lmbda_lpips * out['perceptual_loss'] 

        out['distortion'] = out["mse_loss_scaled"] + (self.beta*out['perceptual_loss_scaled'])

        out["loss"] = (self.lmbda_rate*out["bpp_loss"]) + out['distortion']

        return out
        

        
# class RateDistortionLoss(nn.Module):
#     """Custom rate distortion loss with a Lagrangian parameter."""

#     def __init__(self, lmbda=1e-2):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda

#     def forward(self, output, target):
#         N, _, H, W = target.size()
#         out = {}
#         num_pixels = N * H * W

#         out["bpp_loss"] = sum(
#             (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
#             for likelihoods in output["likelihoods"].values()
#         )
#         out["mse_loss"] = self.mse(output["x_hat"], target)
#         out["loss"] = self.lmbda * 255 ** 2 * out["mse_loss"] + out["bpp_loss"]

#         return out
    
