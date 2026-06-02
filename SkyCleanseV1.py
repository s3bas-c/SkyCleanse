import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpatialAttentionPooling(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, 1)
        )

    def forward(self, x):
        weights = self.attn(x)
        weights = torch.sigmoid(weights)

        x_weighted = x * weights

        out = x_weighted.sum(dim=(2, 3)) / (weights.sum(dim=(2, 3)) + 1e-6)

        return out
    
class SkyCleanse_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, 7, 1, padding=3),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            nn.Conv2d(32, 48, 7, 2, padding=3),
            nn.BatchNorm2d(48),
            nn.SiLU(),

            nn.Conv2d(48, 64, 5, 1, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 64, 5, 2, padding=2),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.final = nn.Linear(64, 1)
        self.pool = SpatialAttentionPooling(64)

    def forward(self, x):
        x = self.layers(x)
        scores = self.pool(x)
        scores = self.final(scores)
        return scores

model = SkyCleanse_v1()
model = model.to(device)
model.load_state_dict(torch.load("skycleanse_v1.pth"))
model.eval()

def inference(np_image):
    #img = cv2.imread("input1.png", cv2.IMREAD_GRAYSCALE)

    #img = img.astype(np.float32)
    np_image = (np_image - np_image.min()) / (np_image.max() - np_image.min() + 1e-8)
    
    img = torch.from_numpy(np_image).float()

    img = img.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).squeeze(0)

    total = (((1 - pred.item())) * 10)

    print(f"--- IMAGE EVALUATION ---> {round(total, 1)}/10")
    return round(total, 1)
    
#inference()
