import argparse
# from comp.zoo import models
# from custom_comp.zoo import models



def int2bool(i):
    i = int(i)
    assert i == 0 or i == 1
    return i == 1


def list_of_float(arg):
    return list(map(float, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="stf",
        help="Model architecture (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=30,
        help="Dataloaders threads (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--lambda",
    #     dest="lmbda",
    #     type=float,
    #     default=1e-2,
    #     help="Bit-rate distortion parameter (default: %(default)s)",
    # )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        default=1e-3,
        type=float,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(256, 256),
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    # parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--cuda", type=int2bool, default=1) # 1 == True

    parser.add_argument("--save", type=int2bool, default=1) # 1 == True
    
    parser.add_argument(
        "--seed", type=int, help="Set random seed for reproducibility", default=42
    )
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint", 
                        default=None)



    parser.add_argument("--resume-train", type=int2bool, default=0) # 1 == True

    parser.add_argument("--save-dir", type = str, help = "Save directory", default = "./exp")
    parser.add_argument("--test-dir", type = str, help = "Kodak Test directory", default = "/data/kodak/")

    parser.add_argument("--lora", type=int2bool, default=0) # 1 == True
    parser.add_argument("--vanilla-adapt", type=int2bool, default=0) # 1 == True


    parser.add_argument("--adapter-config", type = str, default='../configs/lora_8_8.yml')

    parser.add_argument("--adapter-opt", type = str, default='adam', choices=['adam','sgd'])
    parser.add_argument("--adapter-sched", type = str, default='cosine', choices=['lr_plateau','cosine'])



    parser.add_argument("--lambda",dest="lmbda", type=list_of_float, default =  "0.0018,0.0483") 
    
    parser.add_argument("--alpha-perc", type=list_of_float, default = "1,0.95,0.9,0.8,0.6,0.4,0.2")

    parser.add_argument("--compressai-checkpoint", type=int2bool, default=0, help="1 if you start from compressai chekpoint (cheng2020)") # 1 == True
    
    parser.add_argument("--compressai-quality", type=int, default=6)

    parser.add_argument("--conv-adapt", type=int2bool, default=0) # 1 == True
    parser.add_argument("--mixed-adapt", type=int2bool, default=0) # 1 == True

    # parser.add_argument("--unfreeze-hyp", action="store_true", default=False)
    # parser.add_argument("--class-type", type = str, default='natural', choices=["sketch","watercolor","comic","infographics", "clipart", "natural"]) 
    
    parser.add_argument("--linear-alpha", type=int2bool, default=0) # 1 == True

    
    parser.add_argument("--loss-type", type = str, default='mse', choices=["mse","ms-ssim","lpips","rdp"]) 

    parser.add_argument("--lmbda-percpetion", type = float, default=1.0) 

    parser.add_argument("--fixed-lmbda", type=float, default=None) # 1 == True



    parser.add_argument("--compress-often", type=int2bool, default=1) # 1 == True

    parser.add_argument("--inverse-lmbda", type=int2bool, default=0) # 1 == True

    parser.add_argument("--adapted-checkpoint", type=int2bool, default=0) # 1 == True
    parser.add_argument("--adapter-checkpoint-config", type = str, default='../configs/lora_8_8.yml')




    # rate distortion perception
    parser.add_argument("--beta-perc", type=list_of_float, default = "0.01,0.02,0.04,0.05,0.07,0.1,0.15,0.2,0.3,0.4,0.5,1.0")
    parser.add_argument("--adapter-adapter-config", type = str, default='../configs/adapt_lora/lora_8_8.yml')
    parser.add_argument("--beta-max", type=float, default = 5.12)


    args = parser.parse_args()
    return args