import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NVIDIA(nn.Module):
    def __init__(self, no_outputs, use_rgb=False, use_stacked=False, use_disp=False, 
            use_depth=False, use_flow=False, use_speed=False):
        super(NVIDIA, self).__init__()
        self.no_outputs = no_outputs
        self.use_rgb = use_rgb
        self.use_stacked = use_stacked
        self.use_disp = use_disp
        self.use_depth = use_depth
        self.use_flow = use_flow
        self.use_speed = use_speed

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # RGB img
        self.input_channels = 0
        if self.use_rgb:
            self.input_channels += 6 if self.use_stacked else 3
        if self.use_depth or self.use_disp:
            self.input_channels += 1
        if self.use_flow:
            self.input_channels += 2

        self.features = nn.Sequential(
            nn.Conv2d(self.input_channels, 24, (5, 5), padding=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(24, 36, (5, 5), padding=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(36, 48, (5, 5), padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(48, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, (3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 16 + (1 if self.use_speed else 0), 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, self.no_outputs)
        )


    def forward(self, data):
        B, _, H, W = data["img"].shape
        input = []

        mean_rgb = torch.tensor([0.47, 0.44, 0.45]).view(1, 3, 1, 1).to(self.device)
        std_rgb = torch.tensor([0.22, 0.22, 0.22]).view(1, 3, 1, 1).to(self.device)

        mean_depth = torch.tensor([20.38]).view(1, 1, 1, 1).to(self.device)
        std_depth = torch.tensor([16.53]).view(1, 1, 1, 1).to(self.device)

        mean_disp = torch.tensor([0.21]).view(1, 1, 1, 1).to(self.device)
        std_disp = torch.tensor([0.16]).view(1, 1, 1, 1).to(self.device)

        mean_flow = torch.tensor([-2.80, -2.74]).view(1, 2, 1, 1).to(self.device)
        std_flow = torch.tensor([20.01, 10.38]).view(1, 2, 1, 1).to(self.device)

        if self.use_rgb:
            img = data["img"]
            img = (img - mean_rgb) / std_rgb
            input.append(img)

            if self.use_stacked:
                prev_img = data["prev_img"]
                prev_img = (prev_img - mean_rgb) / std_rgb
                input.append(prev_img)

        disp = None
        if self.use_disp:
            disp = data["disp"]
            disp = (disp - mean_disp) / std_disp
            input.append(disp)

        depth = None
        if self.use_depth:
            depth = data["depth"]
            depth = (depth - mean_depth) / std_depth
            input.append(depth)

        flow = None
        if self.use_flow:
            flow = data["flow"]
            flow = (flow - mean_flow) / std_flow
            input.append(flow)

        input = torch.cat(input, dim=1)
        input = self.features(input)
        input = input.reshape(input.shape[0], -1)

        if self.use_speed:
            input = torch.cat((input, data["speed"]), dim=1)
        
        output = self.classifier(input)
        return output, disp, depth, flow
