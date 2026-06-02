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

class SC_dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = torch.from_numpy(np.load("astro_X.npy")).float()
        self.targets = torch.from_numpy(np.load("astro_Y.npy")).float()

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        x = self.inputs[index]
        x = x.unsqueeze(0)
        return x, self.targets[index]
    
dataset = SC_dataset()
dataloader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=True
)

model = SkyCleanse_v1()
model = model.to(device)
learning_rate = 0.0007
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 50


def train():
    best_loss = 1000
    model.train()
    for e in range(epochs):     
        total_loss = 0
        div = 0
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            pred = model(x)
            loss = F.huber_loss(pred, y)
            loss.backward()

            optimizer.step()
            total_loss += loss.item()
            div += 1
        if (total_loss / div) < best_loss:
            best_loss = total_loss / div
            torch.save(model.state_dict(), "skycleanse_v1.pth")
        print(f"{e+1}. loss, {total_loss / div}")

#train()

def inference():
    model.load_state_dict(torch.load("skycleanse_v1.pth"))
    model.eval()

    img = cv2.imread("input1.png", cv2.IMREAD_GRAYSCALE)

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    img = torch.from_numpy(img).float()

    img = img.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img).squeeze(0)

    total = (((1 - pred.item()) ** 1.5) * 10)

    print(f"--- IMAGE EVALUATION ---> {round(total, 1)}/10")
    
inference()
