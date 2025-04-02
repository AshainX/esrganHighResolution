import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ---- STEP 1: DEVICE SETUP (Mac MPS + CPU) ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---- STEP 2: CUSTOM DATASET ----
class CustomDataset(Dataset):
    def __init__(self, hqFolder, lqFolder, lqSize=(120, 120), hqSize=(200, 200)):  
        self.hqImages = sorted(glob.glob(os.path.join(hqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(hqFolder, "**", "*.jpg"), recursive=True))
        self.lqImages = sorted(glob.glob(os.path.join(lqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(lqFolder, "**", "*.jpg"), recursive=True))
        assert len(self.hqImages) == len(self.lqImages), "Mismatch between HQ and LQ image counts"

        self.hqTransform = transforms.Compose([
            transforms.Resize(hqSize),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1,1]
        ])
        self.lqTransform = transforms.Compose([
            transforms.Resize(lqSize),  
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.hqImages)

    def __getitem__(self, index):
        hqImg = Image.open(self.hqImages[index]).convert('RGB')
        lqImg = Image.open(self.lqImages[index]).convert('RGB')
        return self.lqTransform(lqImg), self.hqTransform(hqImg)

# Set dataset paths (Mac-friendly format)
HQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT"
LQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT_LOWRES"

trainDataset = CustomDataset(HQ_FOLDER, LQ_FOLDER)  
trainLoader = DataLoader(trainDataset, batch_size=16, shuffle=True)

# ---- STEP 3: DEFINE Real-ESRGAN MODEL ----
class RRDB(nn.Module):
    def __init__(self, inChannels):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(inChannels, inChannels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x + residual

class RealESRGAN(nn.Module):
    def __init__(self, numRRDB=10, upscaleFactor=200 // 120):
        super(RealESRGAN, self).__init__()
        self.inputConv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.rrdbBlocks = nn.Sequential(*[RRDB(32) for _ in range(numRRDB)])
        self.outputConv = nn.Conv2d(32, 3 * (upscaleFactor ** 2), kernel_size=3, padding=1)
        self.pixelShuffle = nn.PixelShuffle(upscaleFactor)

    def forward(self, x):
        x = self.inputConv(x)
        x = self.rrdbBlocks(x)
        x = self.outputConv(x)
        x = self.pixelShuffle(x)
        return x

# Initialize model
model = RealESRGAN().to(device)

# ---- STEP 4: DEFINE LOSS & OPTIMIZER ----
class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, preds, targets):
        return torch.mean(torch.sqrt((preds - targets) ** 2 + self.epsilon ** 2))

# Instantiate loss functions & optimizer
charbLoss = CharbonnierLoss()
ssimLoss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# ---- STEP 5: TRAINING LOOP ----
numEpochs =  20 
for epoch in range(numEpochs):
    model.train()
    epochLoss = 0

    for lrImgs, hrImgs in tqdm(trainLoader):
        lrImgs, hrImgs = lrImgs.to(device), hrImgs.to(device)

        optimizer.zero_grad()
        preds = model(lrImgs)

        # Ensure predictions match HR image size
        preds = F.interpolate(preds, size=(200, 200), mode='bilinear', align_corners=False)

        # Compute losses
        loss1 = charbLoss(preds, hrImgs)
        loss2 = (1 - ssimLoss(preds, hrImgs))  # Scaled SSIM loss
        loss = loss1 + loss2

        loss.backward()
        optimizer.step()

        epochLoss += loss.item()

    print(f"Epoch {epoch+1}/{numEpochs} - Loss: {epochLoss:.4f}")

# Save trained model
torch.save(model.state_dict(), "RealESRGAN_Mac.pth")
print("Model saved!")