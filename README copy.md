# CALICE: Continuous bitrate control with Adapted LIC modEl

<div align="center">
<img src="./assets/teaser.png" alt="teaser" width="1000"/>
</div>
<div align="center">
<b>Authors:</b> Gabriele Spadaro<sup>1,2</sup>, Alberto Presta<sup>1</sup>, Jhony H. Giraldo<sup>2</sup>, Attilio Fiandrotti<sup>1,2</sup>, Marco Grangetto<sup>1</sup> and Enzo Tartaglione<sup>2</sup>

<sup>1</sup>University of Turin, Italy<br>
<sup>2</sup>LTCI, Télécom Paris, Institut Polytechnique de Paris
</div>

## 📢 Announcement 
<div style="text-align: justify;">
Official pytorch implementation of the paper "<strong>CALICE: Continuous bitrate control with Adapted LIC modEl</strong>", published at <strong>ACM Transactions on Multimedia Computing, Communications, and Applications</strong>. 
<br><br>
  <a href="https://dl.acm.org/doi/abs/10.1145/3820658">
    📄 Read the Journal Paper
  </a>
</div>


## Abstract
Learned image compression (LIC) has drawn much attention recently as it outperforms standardized codecs
in rate-distortion (RD) efficiency. However, a LIC model is typically trained for a specific RD trade-off, and achieving a different target rate requires retraining the model and storing the weights as a whole, limiting the practical applicability of LIC. In this paper, we introduce CALICE, a framework for achieving continuous bitrate control by plugging into a pre-trained LIC model a set of modular adapters. Unlike similar methods that require a distinct set of adapters for each target rate, our method achieves continuous bitrate control by modulating a single set of adapters via a scalar parameter 𝜶, with a total overhead of less than 0.35% of the parameters of the LIC model. This design enables efficient support for multiple distortion objectives by learning lightweight, distortion-aware adapters. We also extend our strategy beyond rate control, demonstrating its ability to provide fine-grained adaptation of perceptual quality along the distortion-perception trade-off. To our knowledge, this is the first method that jointly addresses rate and perceptual control using a unified, low-cost strategy.


<div align="center">
<img src="./assets/image.png" alt="arch" width="1000"/>
</div>


# Trained Models
Results on kodak are saved in ./assets folder in json format

You can download our pretrained variable rate models from [here](https://drive.google.com/drive/folders/1NEDY7TcFWy_LuUr3jIDuHBcW4LHBYqXx?usp=drive_link)

- Anchor models:
--> STF: https://drive.google.com/file/d/1Tj-xWSngX1P1Pg3cPPk74XpIcTcVOQkP/view?usp=sharing
--> TCM: https://drive.google.com/file/d/1KllicXKrzzhaFL73597CYeqrDMs78WXP/view?usp=sharing

- Variable STF models (w/ CALICE):
--> MSE: https://drive.google.com/file/d/1yelw1NKArXzNqO8VWclCN7vy_e-bR3P4/view?usp=sharing
--> MS-SSIM: https://drive.google.com/file/d/10cGB9oJUo3765POWomOWZhSVkE7iMTib/view?usp=sharing
--> LPIPS: https://drive.google.com/file/d/1ir0Hzhyig2riJHe6WKELPBskc0aa7AsN/view?usp=sharing
--> Variable **Rate-Distortion-Perception**: https://drive.google.com/file/d/1gBVKXQpoJJg5nCo3pTcxsBclBCvevpHB/view?usp=sharing

- Variable TCM models (w/ CALICE):
--> MSE: https://drive.google.com/file/d/1aBQNKzM_GQ21y4XOHn74NaG6R-iAAXbw/view?usp=sharing

- Variable Cheng models (w/ CALICE):
--> MSE: https://drive.google.com/file/d/1lJSTOlQhx9IiB4T40jWKIbMB2pCQtVE0/view?usp=sharing


# Dataset 
To download the dataset, we follow the instruction reported in the [QRAF](https://github.com/bytedance/QRAF) repository:


Using Trainningdataset_Preprocessing.py to select the largest 8000 images from [imageNet](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar) and 584 images from [CLIC2020](https://data.vision.ee.ethz.ch/cvl/clic/professional_train_2020.zip) and to preprocess the images as the training dataset.

# Environment 
To create the running environment, please refer to the [Docker file](Dockerfile)

# Variable Rate-Distortion Model 

<div align="center">
<img src="./assets/variable_rate.png" alt="complexity" width="1000"/>
</div>


## Evaluate
eval **STF** on kodak 
```
python -m evaluate.eval_continous \
--test-dir /scratch/dataset/kodak/ \
--save-path res_vr_stf \
--model stf \
--ckpt ../checkpoints/results/stf/mse/checkpoint.pth.tar \
--label CALICE-STF \ 
--adapter-config ../configs/stf_8_8_all.yaml
```

For **TCM** on kodak change:
- --save-path res_vr_tcm
- --model tcm
- --ckpt ../checkpoints/results/tcm/mse/checkpoint.pth.tar
- --label CALICE-TCM
- --adapter-config ../configs/tcm_8_1_all.yaml

For **Cheng** on kodak change:
- --save-path res_vr_cheng
- --model cheng-attn
- --ckpt ../checkpoints/results/cheng_attn/mse/checkpoint.pth.tar
- --label CALICE-Cheng
- --adapter-config ../configs/cheng_8_1_all.yaml

## Training 
Plug and fine-tune modular adapter for obtaining variable rate behavior.
### STF
```
python train.py \
--checkpoint ../checkpoints/anchors/stf_0483_best.pth.tar \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.00001 \
--mixed-adapt 0 --lora 1 --conv-adapt 0 \
--adapter-config ../configs/stf_8_8_all.yaml \
--adapter-opt adam \
--adapter-sched cosine \
--model stf \
--lambda 0.0018,0.0483 \
--save 1 \
--save-dir results/stf/mse/variable_stf
```

### TCM
```
python train.py \
--checkpoint ../checkpoints/anchors/tcm_0.05.pth.tar \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.0001 \
--mixed-adapt 1 --conv-adapt 0 --lora 0 \
--adapter-config ../configs/tcm_8_1_all.yaml \
--adapter-opt adam \ 
--adapter-sched cosine \
--model tcm \
--lambda 0.0025,0.05 \
--save 1 \
--save-dir results/tcm/mse/variable_tcm 
```

### Cheng
```
python train.py \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.0001 \
--mixed-adapt 0 --conv-adapt 1 --lora 0 \
--adapter-config ../configs/cheng_8_1_all.yaml \
--adapter-opt adam \
--adapter-sched cosine \
--model cheng-attn \
--lambda 0.0018,0.0483 \
--compressai-checkpoint 1 \
--compress-often 0 \
--save 1 \
--save-dir results/cheng_attn/mse/variable_cheng
```


# Variable Rate-Distortion-Perception Model
<div align="center">
<img src="./assets/variable_rdp.png" alt="complexity" width="1000"/>
</div>

## Evaluate
```
python -m evaluate.eval_continous_rdp --test-dir /scratch/dataset/kodak/ --save-path ../check_repo/test_rdp_stf --model stf --ckpt ../checkpoints/results/stf/rdp/_checkpoint.pth.tar --label beta-CALICE (STF) --adapter-config ../configs/stf_8_8_all.yaml --adapter-adapter-config ../configs/adapt_stf_8_8_all.yaml
```

## Training
We take our variable rate model and we train a new set of modular adapter to obtain a variable rate-distortion-perception model.

```
python train_perception.py \
--checkpoint results/stf/mse/variable_stf_seed_42/checkpoint.pth.tar \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.00001 \
--adapter-config ../configs/stf_8_8_all.yaml \
--adapter-adapter-config ../configs/adapt_stf_8_8_all.yaml \
--adapter-opt adam \
--adapter-sched cosine \
--model stf \
--loss-type rdp \
--lambda 0.048,1.28 \
--save 1 \
--save-dir results/stf/rdp/variable_rdp_stf 
```


# Distortion-aware variable rate model
<div align="center">
<img src="./assets/distortion_aware.png" alt="complexity" width="1000"/>
</div>

## Evaluate
```
python -m evaluate.eval_continous --test-dir /scratch/dataset/kodak/ --save-path res_vr_stf_mssim --model stf --ckpt ../checkpoints/results/stf/mssim/checkpoint.pth.tar --label CALICE-STF-MSSIM --adapter-config ../configs/stf_8_8_all.yaml
```
For LPIPS change:
- --save-path res_vr_stf_lpips
- --ckpt ../checkpoints/results/stf/lpips_01/checkpoint.pth.tar
- --label CALICE-STF-LPIPS

## Training
example with STF for MS-SSIM

### Step 1
Train a set of adapter to adapt the anchor model on the new distortion function (e.g. MS-SSIM)
```
python train.py \
--checkpoint ../checkpoints/anchors/stf_0483_best.pth.tar \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.00001 \
--mixed-adapt 0 --lora 1 --conv-adapt 0 \
--adapter-config ../configs/stf_8_8_all.yaml \ 
--adapter-opt adam \
--adapter-sched cosine \
--model stf \
--fixed-lmbda 60.50 --loss-type ms-ssim \
--save 1 \
--save-dir results/sft/mssim/1st_step
```

for LPIPS change:
- --fixed-lmbda 0.048 
- --loss-type lpips 
- --lmbda-percpetion 0.1

### Step 2
On this new anchor model we train a set of modular adapter to obtain a variable rate behavior w.r.t the MS-SSIM
```
python train.py \
--checkpoint results/sft/mssim/1st_step_seed_42/checkpoint.pth.tar \
--adapted-checkpoint 1 \
--adapter-checkpoint-config ../configs/stf_8_8_all.yaml \
--dataset /home/ids/gspadaro/data/dataset/qvrf_dataset/ \
--test-dir /home/ids/gspadaro/data/kodak/ \
--epochs 500 \
--learning-rate 0.00001 \
--mixed-adapt 0 --lora 1 --conv-adapt 0 \ 
--adapter-config ../configs/stf_8_8_all.yaml \
--adapter-opt adam \
--adapter-sched cosine \
--model stf \
--lambda 2.40,60.50 \
--loss-type ms-ssim \
--save 1 \
--save-dir results/sft/mssim/2st_step
```

for LPIPS change:
- --checkpoint /path/to/anchor/lpips
- --lambda 0.048,1.28 
- --inverse-lmbda 1 
- --loss-type lpips 
- --lmbda-percpetion 0.1



