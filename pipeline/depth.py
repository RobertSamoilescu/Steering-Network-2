import numpy as np
import PIL.Image as pil
import os
import matplotlib.pyplot as plt

# monodepth & flow (the same architecure)
from .monodepth_dir.depth_decoder import *
from .monodepth_dir.layers import *
from .monodepth_dir.pose_cnn import *
from .monodepth_dir.pose_decoder import *
from .monodepth_dir.resnet_encoder import *
from .monodepth_dir.inverse_warp import *


class Monodepth(object):
    def __init__(self, root_dir: str="./pipeline"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.root_dir = root_dir
        self.intrinsic = np.array([
            [0.61, 0, 0.5],   # width
            [0, 1.22, 0.5],   # height
            [0, 0, 1]],
        dtype=np.float32)
        self.CAM_HEIGHT = 1.5
        
        encoder_path = os.path.join(root_dir, "models", "monodepth", "encoder.pth")
        depth_decoder_path = os.path.join(root_dir, "models", "monodepth", "depth.pth")

        # LOADING PRETRAINED MODEL
        self.encoder = ResnetEncoder(18, False)
        self.encoder = self.encoder.to(self.device)
        
        self.depth_decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, scales=range(4))
        self.depth_decoder = self.depth_decoder.to(self.device)
        
        loaded_dict_enc = torch.load(encoder_path, map_location='cpu')
        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)

        loaded_dict = torch.load(depth_decoder_path, map_location='cpu')
        self.depth_decoder.load_state_dict(loaded_dict)

        self.encoder.eval()
        self.depth_decoder.eval();
        
        
    def get_disp(self, img: torch.tensor):
        """
        @param img: input image (RGB), [B, 3, H, W]
        :returns depth map[B, 1, H, W]
        """
        # normalize
        if img.max() > 1:
            img = img / 255.
        
        img = img.to(self.device)
        
        with torch.no_grad():
            features = self.encoder(img)
            outputs = self.depth_decoder(features)

        disp = outputs[("disp", 0)]
        return disp
    
    def get_depth(self, disp: torch.tensor):
        """
        @param disp: disparity map, [B, 1, H, W]
        :returns depth map
        """
        scaled_disp, depth_pred = disp_to_depth(disp.cpu(), 0.1, 100.0)
        factor = self.get_factor(depth_pred)
        depth_pred *= factor
        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)
        return depth_pred.to(self.device)
    
    def get_factor(self, depth: torch.tensor):
        """
        @param disp: depth map, [B, 1, H, W]
        :returns depth factor
        """
        batch_size, _, height, width = depth.shape
        
        # construct intrinsic camera matrix
        intrinsic = self.intrinsic.copy()
        intrinsic[0, :] *= width
        intrinsic[1, :] *= height
        intrinsic = torch.tensor(intrinsic).repeat(batch_size, 1, 1)

        # get camera coordinates
        cam_coords = pixel2cam(depth.squeeze(1), intrinsic.inverse())
        
        # get some samples from the ground, center of the image
        samples = cam_coords[:, 1, height-10:height, width//2 - 50:width//2 + 50]
        samples = samples.reshape(samples.shape[0], -1)
        
        # get the median
        median = samples.median(1)[0]
  
        # get depth factor
        factor = self.CAM_HEIGHT / median
        factor = factor.reshape(*factor.shape, 1, 1, 1)
        return factor