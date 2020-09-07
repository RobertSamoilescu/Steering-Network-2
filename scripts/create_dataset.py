#!/usr/bin/env python
# coding: utf-8
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from util.JSONReader import *
from util.vis_flow import *

import argparse
import pickle as pkl
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

from pipeline.flow import *
from pipeline.depth import *

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str, help="videos dataset directory")
parser.add_argument("--dst_dir", type=str, help="destination dirctory")
parser.add_argument("--camera_idx", action="store_true", help="if scenes contains camera index")
parser.add_argument("--width", type=int, help="image width", default=256)
parser.add_argument("--height", type=int, help="image height", default=128)
args = parser.parse_args()

# load monodepth and flow
depth_net = Monodepth()
flow_net = Flow()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if not os.path.exists(args.dst_dir):
	os.mkdir(args.dst_dir)
	os.makedirs(os.path.join(args.dst_dir, "img"))
	os.makedirs(os.path.join(args.dst_dir, "disp"))
	os.makedirs(os.path.join(args.dst_dir, "depth"))
	os.makedirs(os.path.join(args.dst_dir, "flow"))
	os.makedirs(os.path.join(args.dst_dir, "data"))


def read_json(root_dir: str, json: str, verbose: bool = False):
    json_reader = JSONReader(root_dir, json, frame_rate=3)
    predicted_course = 0.0

    # get first frame of the video
    prev_frame, _, _ = json_reader.get_next_image()
    prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB)
    prev_frame = cv2.resize(prev_frame[:320, ...], (args.width, args.height))
    frame_idx = 0
    
    while True:
        # get next frame corresponding to current prediction
        frame, speed, rel_course = json_reader.get_next_image()
        if frame.size == 0:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         
        if frame.max() > 1:
            frame = frame / 255.
        if prev_frame.max() > 1:
            prev_frame = prev_frame / 255.

        # process frame
        orig_frame = frame.copy()
        frame = frame[:320, ...]
       	frame = cv2.resize(frame, (args.width, args.height))

        # get depth
        tframe = torch.tensor(frame.transpose(2, 0, 1)).unsqueeze(0).float()
        tframe = F.interpolate(tframe, (256, 512)).to(device)
        tdisp = depth_net.get_disp(tframe)
        tdepth = depth_net.get_depth(tdisp)
        tdisp = F.interpolate(tdisp, (args.height, args.width))
        tdepth = F.interpolate(tdepth, (args.height, args.width))
        disp_map = tdisp.squeeze().cpu().numpy()
        depth_map = tdepth.squeeze().cpu().numpy()

        # get flow
        tprev_frame = torch.tensor(prev_frame.transpose(2, 0, 1)).unsqueeze(0).float()
        tprev_frame = F.interpolate(tprev_frame, (256, 512)).to(device)
        tflow = flow_net.get_flow(tprev_frame, tframe, 1)
        tflow = F.interpolate(tflow, (args.height, args.width))
        flow_map = tflow.squeeze(0).cpu().numpy().transpose(1, 2, 0)

        # save image and data
        scene = json[:-5]

        # save RGB image
        frame_path = os.path.join(args.dst_dir, "img", scene + "." + str(frame_idx) + ".png")
        frame = (255 * frame).astype(np.uint8)
        cv2.imwrite(frame_path, frame[..., ::-1])
       
        # save disp
        disp_path = os.path.join(args.dst_dir, "disp", scene + "." + str(frame_idx) + ".pkl")
        with open(disp_path, "wb") as fout:
         	pkl.dump(disp_map, fout)

        # save depth
        depth_path = os.path.join(args.dst_dir, "depth", scene + "." + str(frame_idx) + ".pkl")
        with open(depth_path, "wb") as fout:
         	pkl.dump(depth_map, fout)

        # save flow
        flow_path = os.path.join(args.dst_dir, "flow", scene + "." + str(frame_idx) + ".pkl")
        with open(flow_path, "wb") as fout:
        	pkl.dump(flow_map, fout)

       	# save speed and relative course
        data_path = os.path.join(args.dst_dir, "data", scene + "." + str(frame_idx) + ".pkl")
        with open(data_path, "wb") as fout:
            pkl.dump({"speed": speed, "rel_course": rel_course}, fout)
        
        frame_idx += 1
        prev_frame = frame
        
        if verbose == True:
            print("Speed: %.2f, Relative Course: %.2f" % (speed, rel_course))
            print("Course: %.2f" % (location['course']))
            # print("Frame shape:", frame.shape)
            # print("Disp shape", disp_map.shape)
            # print("Depth shape", depth_map.shape)
            # print("Flow shape", flow_map.shape)

            disp_vmax = np.percentile(disp_map, 95)
            depth_vmax = np.percentile(depth_map, 95)

            fig, ax = plt.subplots(2, 2)
            ax[0][0].imshow(frame)
            ax[0][1].imshow(disp_map, cmap='magma', vmax=disp_vmax)
            ax[1][0].imshow(depth_map, cmap='gray_r', vmax=depth_vmax)
            ax[1][1].imshow(flow_to_color(flow_map))
            plt.show()


files = os.listdir(args.src_dir)
jsons = [file for file in files if file.endswith(".json")]

for json in tqdm(jsons):
    read_json(args.src_dir, json, False)
