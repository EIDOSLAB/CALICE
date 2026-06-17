from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile

import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torchvision.transforms.functional as F
import numpy as np

import torch
# from .utils import seed_all
from torch.utils.data import DataLoader
import random
import sys

ImageFile.LOAD_TRUNCATED_IMAGES = True

def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)


class SquarePad:
    def __init__(self, patch_size) -> None:
        self.max_patch = max(patch_size[0], patch_size[1])
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        max_wh = max(max_wh, self.max_patch)
        hp = int((max_wh - w) / 2)+1
        vp = int((max_wh - h) / 2)+1
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')



class TestKodakDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = sorted(glob(os.path.join(self.data_dir, "*.*")))

        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        
        return self.transform(image)

    def __len__(self):
        return len(self.image_path)
    
