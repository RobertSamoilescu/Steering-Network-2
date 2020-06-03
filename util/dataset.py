import os
import pandas as pd
import numpy as np
import pickle as pkl
import PIL.Image as pil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


def gaussian_dist(mean=200.0, std=5, nbins=401):
    x = np.arange(401)
    pdf = np.exp(-0.5 * ((x - mean) / std)**2)
    pmf = pdf / pdf.sum()
    return pmf


def normalize(img):
    return img / 255

def unnormalize(img):
    return (255 * img).astype(np.uint8)


class UPBDataset(Dataset):
    def __init__(self, root_dir: str, train: bool=True):
        path = os.path.join(root_dir, "train.csv" if train else "test.csv")
        self.files = list(pd.read_csv(path)["name"])
        self.train = train

        self.img  = [os.path.join(root_dir, "img", file   + ".png") for file in self.files]
        self.data = [os.path.join(root_dir, "data", file  + ".pkl") for file in self.files]
        self.disp = [os.path.join(root_dir, "disp", file  + ".pkl") for file in self.files]
        self.depth= [os.path.join(root_dir, "depth", file + ".pkl") for file in self.files]
        self.flow = [os.path.join(root_dir, "flow", file + ".pkl")  for file in self.files]

        self.prev_img = []
        for file in self.files:
            scene, idx = file.split(".")
            self.prev_img.append(os.path.join(root_dir, "img", ".".join([scene, str(int(idx) - 1), "png"])))

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        do_aug = np.random.rand() > 0.5

        if do_aug and self.train:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        # read image
        img = pil.open(self.img[idx])
        prev_img = pil.open(self.prev_img[idx])

        # perform color augmentation
        img = color_aug(img)
        prev_img = color_aug(prev_img)

        # transform images to numpy arrays
        img = np.asarray(img)
        prev_img = np.asarray(prev_img)
        
        # transpose C, H, W
        img = img.transpose(2, 0, 1)
        prev_img = prev_img.transpose(2, 0, 1)

        # normalize images in [0, 1]
        img = normalize(img)
        prev_img = normalize(prev_img)

        # read data
        with open(self.data[idx], "rb") as fin:
            data = pkl.load(fin)

        # target
        data['rel_course'] = np.clip(data['rel_course'], -20, 20)
        pmf = gaussian_dist(mean=10 * data['rel_course'] + 200.)

        # read disp
        with open(self.disp[idx], "rb") as fin:
            disp = pkl.load(fin)

        # read depth
        with open(self.depth[idx], "rb") as fin:
            depth = pkl.load(fin)

        # read flo
        with open(self.flow[idx], "rb") as fin:
            flow = pkl.load(fin)
            flow = flow.transpose(2, 0, 1)

        return {
            "prev_img": torch.tensor(prev_img).float(),
            "img": torch.tensor(img).float(),
            "disp": torch.tensor(disp).unsqueeze(0).float(),
            "depth": torch.tensor(depth).unsqueeze(0).float(),
            "flow": torch.tensor(flow).float(),
            "rel_course": torch.tensor(pmf).float(),
            "rel_course_val": data['rel_course'],
            "speed": torch.tensor(data["speed"]).unsqueeze(0).float()
        }



if __name__ == "__main__":
    x = gaussian_dist()
    sns.lineplot(np.arange(401), x)
    plt.show()
