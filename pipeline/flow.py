from .monodepth_dir.resnet_encoder import *
from .monodepth_dir.pose_decoder import *
from .monodepth_dir.pose_cnn import *
from .monodepth_dir.layers import *
from .monodepth_dir.depth_decoder import *

import pipeline.flow_dir.encoder as ENC
import pipeline.flow_dir.decoder as DEC
import pipeline.flow_dir.layers as LYR 

from .vis import *
from .warp import *

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import PIL.Image as pil



class Flow(object):
    def __init__(self, root_dir: str="./pipeline"):
        self.root_dir = root_dir
        self.intrinsic = np.array([
            [0.61, 0, 0.5, 0],   # width
            [0, 1.22, 0.5, 0],   # height
            [0, 0, 1, 0],
            [0, 0, 0, 1]],
        dtype=np.float32)

        self.WIDTH = 512
        self.HEIGHT = 256
        self.scale = np.array([
            [self.WIDTH, 0, 0, 0],
            [0, self.HEIGHT, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder_path = os.path.join(root_dir, "models", "monodepth", "encoder.pth")
        depth_decoder_path = os.path.join(root_dir, "models", "monodepth", "depth.pth")
        pose_encoder_path = os.path.join(root_dir, "models", "monodepth", "pose_encoder.pth")
        pose_decoder_path = os.path.join(root_dir, "models", "monodepth", "pose.pth", )
        dflow_path = os.path.join(root_dir, "models", "flow", "default.pth")

        # LOADING PRETRAINED DEPTH MODEL
        self.encoder = ResnetEncoder(18, False)
        self.encoder = self.encoder.to(self.device)
        
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder = self.depth_decoder.to(self.device)

        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        # LOADING PRETRAINED POSE MODEL        
        self.pose_encoder = ResnetEncoder(18, False, num_input_images=2)
        self.pose_encoder = self.pose_encoder.to(self.device)

        self.pose_decoder = PoseDecoder(
            self.pose_encoder.num_ch_enc,
            num_input_features=1,
            num_frames_to_predict_for=2
        )
        self.pose_decoder = self.pose_decoder.to(self.device)

        # LOADING PRETRAINDE DYNAMIC FLOW MODEL
        self.dflow_encoder = ENC.ResnetEncoder(
            num_layers=18,
            pretrained=True,
            num_input_images=2
        )
        self.dflow_encoder.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dflow_encoder = self.dflow_encoder.to(self.device)

        self.dflow_decoder = DEC.FlowDecoder(
            num_ch_enc=self.dflow_encoder.num_ch_enc,
            scales=range(4),
            num_output_channels=2,
            use_skips=True
        ).to(self.device)

        # declare ssim
        self.ssim = LYR.SSIM().to(self.device)

        loaded_dict = torch.load(pose_encoder_path, map_location='cpu')
        self.pose_encoder.load_state_dict(loaded_dict)
        
        loaded_dict = torch.load(pose_decoder_path, map_location='cpu')
        self.pose_decoder.load_state_dict(loaded_dict)

        loaded_dict = torch.load(dflow_path, map_location='cpu')
        self.dflow_encoder.load_state_dict(loaded_dict['encoder'])
        self.dflow_decoder.load_state_dict(loaded_dict['decoder'])

        self.encoder.eval()
        self.depth_decoder.eval()
        self.pose_decoder.eval()
        self.dflow_encoder.eval()
        self.dflow_decoder.eval()
        
    def get_pix_coords(self, prev_img: torch.tensor, img: torch.tensor, batch_size=1):
        # get disp
        with torch.no_grad():
            depth_features = self.encoder(img)
            depth_output = self.depth_decoder(depth_features)
            disp = depth_output[("disp", 0)]
            _, depth = disp_to_depth(disp, 0.1, 100.0)

        # get pose
        with torch.no_grad():
            input = torch.cat([prev_img, img], dim=1)
            pose_features = self.pose_encoder(input)
            pose_output = self.pose_decoder([pose_features])
            axisangle, translation = pose_output

        # get transformation between frames
        T = transformation_from_parameters(
            axisangle[:, 0], translation[:, 0], invert=True)
        K = torch.tensor(self.scale @ self.intrinsic)\
            .unsqueeze(0).repeat(batch_size, 1, 1).float()\
            .to(self.device)
        inv_K = K.inverse().float().to(self.device)


        # get camera coords
        backproject_depth = BackprojectDepth(batch_size, self.HEIGHT, self.WIDTH).to(self.device)
        cam_points = backproject_depth(depth, inv_K).float()
        
        # get pixel coords
        project_3d = Project3D(batch_size, self.HEIGHT, self.WIDTH)
        pix_coords = project_3d(cam_points, K, T).float()

        return pix_coords

    def get_rigid_flow(self, pix_coords, batch_size=1):
        # mesh grid 
        W = self.WIDTH
        H = self.HEIGHT
        B = batch_size

        xx = torch.arange(0, W).view(1,-1).repeat(H,1)
        yy = torch.arange(0, H).view(-1,1).repeat(1,W)
        xx = xx.view(1,H,W,1).repeat(B,1,1,1)
        yy = yy.view(1,H,W,1).repeat(B,1,1,1)
        grid = torch.cat((xx,yy),3).float()
        grid = grid.to(self.device)

        return pix_coords - grid


    def get_flow(self, prev_img: torch.tensor, img: torch.tensor, batch_size=1):
        # get rigid flow
        pix_coords = self.get_pix_coords(prev_img, img, batch_size)
        rflow = self.get_rigid_flow(pix_coords, batch_size)
        rflow = rflow.transpose(2, 3).transpose(1, 2)

        # wrap frame using rigid flow
        wimg = warp(prev_img, rflow)

        # comput error map
        ssim_loss = self.ssim(wimg, img).mean(1, True)
        l1_loss = torch.abs(wimg - img).mean(1, True)
        err_map = 0.85 * ssim_loss + 0.15 * l1_loss

        # get dynamic flow correction
        input = torch.cat([prev_img, img, wimg], dim=1)
        with torch.no_grad():
            enc_ouput = self.dflow_encoder(input, rflow, err_map)
            dec_output = self.dflow_decoder(enc_ouput)
            dflow = dec_output[('flow', 0)]

        flow = dflow + rflow
        return flow