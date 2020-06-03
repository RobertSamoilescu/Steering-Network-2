import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import argparse
import os

from util.dataset import *
from util.io import *
from tqdm import tqdm

train_dataset = UPBDataset("./dataset", train=True)
train_dataloader = DataLoader(
	train_dataset, 
	batch_size=12, 
	shuffle=False, 
	drop_last=False, 
	num_workers=6
)

if __name__ == "__main__":
	mean_rgb, std_rgb = torch.zeros(3), torch.zeros(3)
	mean_depth, std_depth = torch.zeros(1), torch.zeros(1)
	mean_disp, std_disp = torch.zeros(1), torch.zeros(1)
	mean_flow, std_flow = torch.zeros(2), torch.zeros(2)
	nb_samples = 0

	for i, data in tqdm(enumerate(train_dataloader)):
		batch_samples = data['img'].size(0)
		rgb = data['img'].view(data['img'].size(0), data['img'].size(1), -1)
		depth = data['depth'].view(data['depth'].size(0), data['depth'].size(1), -1)
		disp = data['disp'].view(data['disp'].size(0), data['disp'].size(1), -1)
		flow = data['flow'].view(data['flow'].size(0), data['flow'].size(1), -1)


		mean_rgb += rgb.mean(2).sum(0)
		std_rgb += rgb.std(2).sum(0)

		mean_depth += depth.mean(2).sum(0)
		std_depth += depth.std(2).sum(0)

		mean_disp += disp.mean(2).sum(0)
		std_disp += disp.std(2).sum(0)

		mean_flow += flow.mean(2).sum(0)
		std_flow += flow.std(2).sum(0)

		nb_samples += batch_samples
		if i > 100:
			break


	mean_rgb, std_rgb = mean_rgb/nb_samples, std_rgb/nb_samples
	mean_depth, std_depth = mean_depth/nb_samples, std_depth/nb_samples
	mean_disp, std_disp = mean_disp/nb_samples, std_disp/nb_samples
	mean_flow, std_flow = mean_flow/nb_samples, std_flow/nb_samples

	opts = {
		"rgb": (mean_rgb, std_rgb),
		"depth": (mean_depth, std_depth),
		"disp": (mean_disp, std_disp),
		"flow": (mean_flow, std_flow)
	}

	print(opts)
