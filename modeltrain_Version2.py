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

# Memory management for MPS - use a valid value between 0 and 1
# if device.type == 'mps':
#     os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.2'  # Fixed value

# ---- STEP 2: CUSTOM DATASET ----
# Define constants for image sizes
LQ_SIZE = (100, 100)  # Low resolution
HQ_SIZE = (200, 200)  # High resolution (2x upscaling)
UPSCALE_FACTOR = 2    # Clean integer scaling factor

class CustomDataset(Dataset):
    def __init__(self, hqFolder, lqFolder, lqSize=LQ_SIZE, hqSize=HQ_SIZE):  
        self.hqImages = sorted(glob.glob(os.path.join(hqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(hqFolder, "**", "*.jpg"), recursive=True))
        self.lqImages = sorted(glob.glob(os.path.join(lqFolder, "**", "*.png"), recursive=True) +
                               glob.glob(os.path.join(lqFolder, "**", "*.jpg"), recursive=True))
        assert len(self.hqImages) == len(self.lqImages), "Mismatch between HQ and LQ image counts"

        self.hqTransform = transforms.Compose([
            transforms.Resize(hqSize),  
            transforms.ToTensor(),
        ])
        self.lqTransform = transforms.Compose([
            transforms.Resize(lqSize),  
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.hqImages)

    def __getitem__(self, index):
        hqImg = Image.open(self.hqImages[index]).convert('RGB')
        lqImg = Image.open(self.lqImages[index]).convert('RGB')
        return self.lqTransform(lqImg), self.hqTransform(hqImg)

# Set dataset paths (keeping your original paths)
HQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT"
LQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT_LOWRES"

# ---- STEP 3: DEFINE MEMORY-EFFICIENT ESRGAN MODEL ----
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual
        return self.relu(out)

class MemoryEfficientESRGAN(nn.Module):
    def __init__(self, num_blocks=6, channels=32, upscale_factor=UPSCALE_FACTOR):
        super(MemoryEfficientESRGAN, self).__init__()
        
        # Initial convolution
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )
        
        # Mid convolution
        self.mid_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        initial_out = self.initial(x)
        residual_out = self.residual_blocks(initial_out)
        mid_out = self.mid_conv(residual_out)
        mid_out = mid_out + initial_out  # Global residual connection
        upsampled = self.upsampling(mid_out)
        final_out = self.final(upsampled)
        return torch.clamp(final_out, 0, 1)  # Ensure output is in [0, 1]

# ---- STEP 4: TRAINING SETUP ----
def train_model(batch_size=2, num_epochs=20, lr=1e-4):
    # Create dataset and dataloader with smaller batch size
    trainDataset = CustomDataset(HQ_FOLDER, LQ_FOLDER)  
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0)  # Set num_workers=0 for Mac
    
    # Initialize model
    model = MemoryEfficientESRGAN().to(device)
    
    # Loss functions and optimizer
    mse_loss = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        with tqdm(trainLoader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for lr_imgs, hr_imgs in pbar:
                # Move to device
                lr_imgs = lr_imgs.to(device)
                hr_imgs = hr_imgs.to(device)
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    sr_imgs = model(lr_imgs)
                    
                    # Calculate loss
                    loss = mse_loss(sr_imgs, hr_imgs)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: Out of memory error. Skipping batch.")
                        # Try to free memory
                        if device.type == 'mps':
                            # Mac MPS specific memory cleanup
                            torch.mps.empty_cache()
                            
                            # If still having issues, try reducing batch size here
                            if batch_size > 1:
                                print(f"Reducing batch size to {batch_size-1}")
                                return train_model(batch_size=batch_size-1, num_epochs=num_epochs, lr=lr)
                        continue
                    else:
                        raise e
        
        # Calculate average loss
        avg_loss = epoch_loss / len(trainLoader)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
        
        # Adjust learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "ESRGAN_best.pth")
            print(f"New best model saved with loss: {best_loss:.6f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pth")
    
    # Save final model
    torch.save(model.state_dict(), "ESRGAN_final.pth")
    print("Training complete! Final model saved.")
    
    return model

# ---- STEP 5: INFERENCE FUNCTION ----
def upscale_image(model_path, img_path, output_path):
    # Load model
    model = MemoryEfficientESRGAN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(LQ_SIZE),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Upscale
    with torch.no_grad():
        output = model(img_tensor)
    
    # Convert to image and save
    output = output.squeeze().cpu().clamp(0, 1)
    output_img = transforms.ToPILImage()(output)
    output_img.save(output_path)
    print(f"Upscaled image saved to {output_path}")

# ---- STEP 6: MAIN CODE ----
if __name__ == "__main__":
    try:
        # Free up memory
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Start training with smaller batch size
        print("Starting training with reduced memory usage...")
        model = train_model(batch_size=2, num_epochs=10, lr=1e-4)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        
        # If still getting memory errors, try these settings
        if "out of memory" in str(e).lower():
            print("\nTrying with more aggressive memory settings...")
            print("Please run again with these settings:")
            print("1. Reduce batch size to 1")
            print("2. Reduce number of residual blocks")
            print("3. Reduce feature channels")
            print("Example: train_model(batch_size=1, num_epochs=20, lr=1e-4)")