import torch
import torch.nn
import torch.nn.functional as F

import os
import argparse

from models.nvidia import *
from models.resnet import *
from util.dataset import *
from util.io import *

from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=12, help="batch size")
parser.add_argument("--vis_dir", type=str, default="./snapshots", help="visualize directory")
parser.add_argument("--dataset_dir", type=str, default="./dataset", help="dataset directory")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
parser.add_argument("--use_rgb", action="store_true", help="use rgb images as input")
parser.add_argument("--use_speed", action="store_true", help="append speed to nvidia model")
parser.add_argument("--use_depth", action="store_true", help="use depth as auxiliary loss for resnet")
parser.add_argument("--use_disp", action="store_true", help="use disparity as auxiliary loss for resnet")
parser.add_argument("--use_flow", action="store_true", help="use flow as auxiliary loss for resnet")
parser.add_argument("--use_balance", action="store_true", help="balance dataset")
parser.add_argument("--use_stacked", action="store_true", help="use stacked frames for resnet")
parser.add_argument("--load_model", type=int, help="checkpoint number", default=None)
parser.add_argument("--model", type=str, help="[nvidia, resnet]", default=None)
args = parser.parse_args()

torch.manual_seed(0)

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
nbins=401
experiment = ""

if args.model == "nvidia":
	model = NVIDIA(
		no_outputs=nbins, 
		use_rgb=args.use_rgb,
		use_stacked=args.use_stacked,
		use_disp=args.use_disp,
		use_depth=args.use_depth,
		use_flow=args.use_flow,
		use_speed=args.use_speed).to(device)
	experiment += "nvidia"
else:
	model = RESNET(
		no_outputs=nbins,
		use_rgb=args.use_rgb,
		use_stacked=args.use_stacked,
		use_disp=args.use_disp,
		use_depth=args.use_depth,
		use_flow=args.use_flow,
		use_speed=args.use_speed).to(device)
	experiment += "resnet"

if args.use_rgb:
	experiment += "_rgb"
if args.use_speed:
	experiment += "_speed"
if args.use_stacked:
	experiment += "_stacked"
if args.use_depth:
	experiment += "_depth"
if args.use_disp:
	experiment += "_disp"
if args.use_flow:
	experiment += "_flow"
if args.use_balance:
	experiment += "_balance"

# load model
path = os.path.join("snapshots", experiment, "ckpts", "default.pth")
load_ckpt(path, [('model', model)])
model.eval()

# define criterion
criterion = nn.KLDivLoss(reduction="none")

# define dataloader
test_dataset = UPBDataset(args.dataset_dir, train=False)
test_dataloader = DataLoader(
	test_dataset,
	batch_size=args.batch_size,
	shuffle=False,
	drop_last=False,
	num_workers=args.num_workers
)

def main():
	losses = []

	with torch.no_grad():
		for i, data in tqdm(enumerate(test_dataloader)):
                        # send data to device
			for key in data:
				data[key] = data[key].to(device)
			# get output
			course_logits, _, _, _ = model(data)

			# compute steering losss
			log_softmax_output = F.log_softmax(course_logits, dim=1)
			kl = criterion(log_softmax_output, data["rel_course"])
			kl = kl.sum(dim=1).cpu().numpy()
			losses += list(kl)

	results = {
		"experiment": experiment,
		"sum": np.sum(losses),
		"mean": np.mean(losses),
		"std": np.std(losses),
		"min": np.min(losses),
		"max": np.max(losses),
		"median": np.median(losses)
	}
	return results

if __name__ == "__main__":
	results = main()
	print(results)

	if not os.path.exists("./results"):
		os.mkdir("./results")

	with open(os.path.join("results", experiment), "wb") as fout:
		pkl.dump(results, fout)
