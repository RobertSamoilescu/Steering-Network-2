import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import matplotlib.pyplot as plt
import argparse
import os

from models.nvidia import *
from models.resnet import *
from util.dataset import *
from util.vis import *
from util.io import *
from util.early import *

from tqdm import tqdm
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
parser.add_argument("--step_size", type=int, default=10, help="scheduler learning rate step size")
parser.add_argument("--batch_size", type=int, default=12, help="batch size")
parser.add_argument("--log_interval", type=int, default=50, help="number of batches to log after")
parser.add_argument("--vis_interval", type=int, default=500, help="number of batches to visualize after")
parser.add_argument("--save_interval", type=int, default=1, help="number of epoch to save the model after")
parser.add_argument("--log_dir", type=str, default="./logs", help="logging directory")
parser.add_argument("--vis_dir", type=str, default="./snapshots", help="visualize directory")
parser.add_argument("--optimizer", type=str, default="adam", help="optimizer available: adam, rmsprop")
parser.add_argument("--dataset_dir", type=str, default="./dataset", help="dataset directory")
parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--num_vis", type=int, default=4, help="number of visualizations")
parser.add_argument("--use_rgb", action="store_true", help="use rgb images as input")
parser.add_argument("--use_speed", action="store_true", help="append speed to nvidia model")
parser.add_argument("--use_depth", action="store_true", help="use depth as auxiliary loss for resnet")
parser.add_argument("--use_disp", action="store_true", help="use disparity as auxiliary loss for resnet")
parser.add_argument("--use_flow", action="store_true", help="use flow as auxiliary loss for resnet")
parser.add_argument("--use_balance", action="store_true", help="balance dataset")
parser.add_argument("--use_stacked", action="store_true", help="use stacked frames for resnet")
parser.add_argument("--load_model", type=str, help="checkpoint name", default=None)
parser.add_argument("--model", type=str, default="nvidia", help="[nvidia, resnet]")
parser.add_argument("--patience", type=int, default=2, help="early stopping patience")
parser.add_argument("--weight_decay", type=float, default=0, help="weight decay optimizer")
args = parser.parse_args()

# set seed
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define model
nbins=401
experiment = ""

if args.model == 'nvidia':
	model = NVIDIA(
		no_outputs=nbins,
		use_rgb=args.use_rgb, 
		use_stacked=args.use_stacked,
		use_disp=args.use_disp,
		use_depth=args.use_depth,
		use_flow=args.use_flow,
		use_speed=args.use_speed
	).to(device)
	experiment += "nvidia"

else:
	model = RESNET(
		no_outputs=nbins,
		use_rgb=args.use_rgb, 
		use_stacked=args.use_stacked,
		use_disp=args.use_disp,
		use_depth=args.use_depth,
		use_flow=args.use_flow,
		use_speed=args.use_speed
	).to(device)
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

# define criterion
criterion = nn.KLDivLoss(reduction="batchmean")

# define optimizer
if args.optimizer == "adam":
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)

# define dataloaders
train_dataset = UPBDataset(args.dataset_dir, train=True)
test_dataset = UPBDataset(args.dataset_dir, train=False)

if args.use_balance:
	weights = pd.read_csv(os.path.join(args.dataset_dir, "weight.csv")).to_numpy().reshape(-1)
	weights = torch.DoubleTensor(weights)
	sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

	train_dataloader = DataLoader(
		train_dataset, 
		batch_size=args.batch_size, 
		sampler=sampler,
		num_workers=args.num_workers,
		pin_memory=True
	)

	experiment += "_balance"
else:
	train_dataloader = DataLoader(
		train_dataset, 
		batch_size=args.batch_size, 
		shuffle=True, 
		drop_last=True, 
		num_workers=args.num_workers
	)

test_dataloader = DataLoader(
	test_dataset,
	batch_size=args.batch_size,
	shuffle=True,
	drop_last=True,
	num_workers=2
)

# create necessary directorirs
if not os.path.exists(args.log_dir):
	os.mkdir(args.log_dir)

if not os.path.exists(args.vis_dir):
	os.mkdir(args.vis_dir)

if not os.path.exists(os.path.join(args.vis_dir, experiment)):
	os.makedirs(os.path.join(args.vis_dir, experiment))
	os.makedirs(os.path.join(args.vis_dir, experiment, "imgs_train"))
	os.makedirs(os.path.join(args.vis_dir, experiment, "imgs_test"))
	os.makedirs(os.path.join(args.vis_dir, experiment, "ckpts"))


# define writer
writer = SummaryWriter(os.path.join(args.log_dir, experiment))

# running loss
rloss = None

def run_epoch(dataloader, epoch, train_flag=True):
	global rloss

	# total loss
	total_loss = 0.0

	for  i, data in enumerate(dataloader):
		# data["prev_img"] = F.interpolate(data["prev_img"], (128, 256))
		# data["img"] = F.interpolate(data["img"], (128, 256))
		# data["disp"] = F.interpolate(data["disp"], (128, 256))
		# data["depth"] = F.interpolate(data["depth"], (128, 256))
		# data["flow"] = F.interpolate(data["flow"], (128, 256))

		if train_flag:
			optimizer.zero_grad()

		# pass data through model
		for key in data:
			data[key] = data[key].to(device)

		if train_flag:
			course_logits, disp, depth, flow = model(data)
		else:
			with torch.no_grad():
				course_logits, disp, depth, flow = model(data)

		# compute steering loss
		# softmax_output = F.softmax(course_logits, dim=1).clamp(min=1e-3)
		# log_softmax_output = torch.log(softmax_output)
		log_softmax_output = F.log_softmax(course_logits, dim=1)
		loss = criterion(log_softmax_output, data["rel_course"])

		# gradient step
		if train_flag:
			loss.backward()
			optimizer.step()

		# update running losses
		rloss = loss.item() if rloss is None else rloss * 0.99 + 0.01 * loss.item() 
		
		# update total loss
		total_loss += loss.item()
		
		# print statistics
		if train_flag and i % args.log_interval == 0:
			print("Epoch: %d, Batch: %d, Running loss: %.4f" % (epoch, i, rloss))
			index = epoch * (len(train_dataset) // args.batch_size) + i
			writer.add_scalar("rloss", rloss, index)			
			writer.flush()

		# visualization
		if i % args.vis_interval == 0:
			num_vis = min(args.num_vis, args.batch_size)
			softmax_output = F.softmax(course_logits, dim=1)

			if args.use_disp:
				disp = disp[:num_vis]
			if args.use_depth:
				depth = depth[:num_vis]
			if args.use_flow:
				flow = flow[:num_vis]
		
			folder = "imgs_train" if train_flag else "imgs_test"
			path = os.path.join(args.vis_dir, experiment, folder, "epoch:%d.batch:%d.png" % (epoch, i))
			visualisation(
				img=data["img"][:num_vis],
				disp=disp,
				depth=depth,
				flow=flow,
				course=data["rel_course"][:num_vis], 
				softmax_output=softmax_output[:num_vis], 
				num_vis=num_vis, 
				path=path
			)
	return total_loss

if __name__ == "__main__":
	start_epoch = 0
	best_score = None

	if args.load_model:
		ckpt_name = os.path.join(args.vis_dir, experiment, "ckpts", str(args.load_model) + ".pth")
		start_epoch = load_ckpt(
				ckpt_name=ckpt_name,
				models=[('model', model)], 
				optimizers=[('optimizer', optimizer)],
				schedulers=[('scheduler', scheduler)],
				rlosses=[('rloss', rloss)], 
				best_scores=[('best_score', best_score)]
		)

	# define early stopping
	early_stopping = EarlyStopping(patience=args.patience)

	for epoch in tqdm(range(start_epoch, args.num_epochs)):
		model.train()
		train_loss = run_epoch(train_dataloader, epoch=epoch, train_flag=True)
		train_loss /= len(train_dataloader)

		model.eval()
		test_loss = run_epoch(test_dataloader, epoch=epoch, train_flag=False)
		test_loss /= len(test_dataloader)

		# log
		print("Epoch: %d, Train Loss: %.4f, Test Loss: %4f" % (epoch, train_loss, test_loss))
		writer.add_scalar("train_loss", train_loss, epoch)
		writer.add_scalar("test_loss", test_loss, epoch)
		writer.flush()

		if epoch % args.save_interval == 0 and (best_score is None or best_score > test_loss):
			best_score = test_loss
			name = "epoch: %d; tloss: %.2f; vloss: %.2f.pth" % (epoch, train_loss, test_loss)
			ckpt_name = os.path.join(args.vis_dir, experiment, "ckpts", name)
			save_ckpt(
				ckpt_name, 
				models=[('model', model)], 
				optimizers=[('optimizer', optimizer)],
				schedulers=[('scheduler', scheduler)],
				rlosses=[('rloss', rloss)], 
				best_scores=[('best_score', best_score)],
				n_iter=epoch + 1
			)
			print("Model saved!")

		# learning rate scheduler step
		scheduler.step()

		# early stopping
		early_stopping(test_loss)
		if early_stopping.early_stop == True:
			break

writer.close()
