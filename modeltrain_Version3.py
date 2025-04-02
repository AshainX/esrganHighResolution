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
import numpy as np
import matplotlib.pyplot as plt

# ----DEVICE SETUP (Mac MPS + CPU) ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants for image sizes
LQ_SIZE = (100, 100)  # Low resolution
HQ_SIZE = (200, 200)  # High resolution (2x upscaling)
UPSCALE_FACTOR = 2    # Clean integer scaling factor

# ---- Custom Dataset ------
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

# Set dataset paths (update these to your actual paths)
HQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT"
LQ_FOLDER = "/Users/ashutosh/Desktop/ElementaryCQT_LOWRES"

# Enhanced Residual Dense Block
class ResidualDenseBlock(nn.Module):
    def __init__(self, channels, growth_rate=32):
        super(ResidualDenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = channels
        
        # More dense connections
        for i in range(4):
            inter_channels = growth_rate * (i + 1)
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1),
                nn.BatchNorm2d(growth_rate),
                nn.ReLU(inplace=True)
            )
            self.layers.append(conv_layer)
            in_channels += growth_rate

        # Final convolution to map back to original channels
        self.final_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            # Concatenate all previous features
            feature_concat = torch.cat(features, dim=1)
            output = layer(feature_concat)
            features.append(output)

        # Final dense connection and convolution
        feature_concat = torch.cat(features, dim=1)
        output = self.final_conv(feature_concat)
        return x + output  # Residual connection

# Enhanced ESRGAN Model
class EnhancedESRGAN(nn.Module):
    def __init__(self, num_blocks=12, channels=64, upscale_factor=2):
        super(EnhancedESRGAN, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        
        # Dense Residual Blocks with more advanced architecture
        self.residual_blocks = nn.Sequential(
            *[ResidualDenseBlock(channels) for _ in range(num_blocks)]
        )
        
        # Global feature fusion
        self.global_fusion = nn.Sequential(
            nn.Conv2d(channels * (num_blocks + 1), channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )
        
        # Advanced upsampling
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.PReLU()
        )
        
        # Final refinement
        self.final = nn.Sequential(
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Tanh()  # Use Tanh for better normalization
        )
        
        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        initial_out = self.initial(x)
        
        # Collect features from each residual block
        features = [initial_out]
        block_out = initial_out
        for block in self.residual_blocks:
            block_out = block(block_out)
            features.append(block_out)
        
        # Global feature fusion
        global_features = torch.cat(features, dim=1)
        fused_features = self.global_fusion(global_features)
        
        # Upsampling and final refinement
        upsampled = self.upsampling(fused_features)
        final_out = self.final(upsampled)
        
        # Rescale to [0, 1]
        return (final_out + 1) / 2

# Enhanced Training Function
def train_model(batch_size=4, num_epochs=30, lr=5e-4):
    # Perceptual Loss Components
    def perceptual_loss(sr_imgs, hr_imgs):
        # Combine multiple loss functions
        mse = F.mse_loss(sr_imgs, hr_imgs)
        l1 = F.l1_loss(sr_imgs, hr_imgs)
        
        # Additional smoothness constraint
        sobel_sr = sobel_edges(sr_imgs)
        sobel_hr = sobel_edges(hr_imgs)
        edge_loss = F.mse_loss(sobel_sr, sobel_hr)
        
        return mse + 0.5 * l1 + 0.2 * edge_loss

    def sobel_edges(x):
        # Sobel operator for edge detection
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
        
        kernel_x = kernel_x.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        kernel_y = kernel_y.view(1, 1, 3, 3).repeat(x.size(1), 1, 1, 1)
        
        edges_x = F.conv2d(x, kernel_x, padding=1, groups=x.size(1))
        edges_y = F.conv2d(x, kernel_y, padding=1, groups=x.size(1))
        
        return torch.sqrt(edges_x**2 + edges_y**2)

    # Create dataset and dataloader
    trainDataset = CustomDataset(HQ_FOLDER, LQ_FOLDER)  
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    
    # Initialize model with more parameters
    model = EnhancedESRGAN(num_blocks=12, channels=64).to(device)
    
    # Advanced optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                             lr=lr, 
                             weight_decay=1e-4,  # Add weight decay for regularization
                             betas=(0.9, 0.999))
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Initial restart period
        T_mult=2,  # Exponential increase in restart period
        eta_min=1e-6  # Minimum learning rate
    )
    
    # Training loop with early stopping and gradient clipping
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
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
                sr_imgs = model(lr_imgs)
                
                # Compute perceptual loss
                loss = perceptual_loss(sr_imgs, hr_imgs)
                
                # Backward pass with gradient clipping
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Update statistics
                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # Learning rate scheduling
            scheduler.step()
            
            # Average loss calculation
            avg_loss = epoch_loss / len(trainLoader)
            print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(model.state_dict(), "ESRGAN_best.pth")
            else:
                patience_counter += 1
            
            # Break if no improvement
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    print("Training complete! Best model saved.")
    return model

# ---- STEP 6: ACCURACY EVALUATION FUNCTION ----
def evaluate_accuracy(model, dataset, batch_size=2, num_samples=None):
    """
    Evaluate the accuracy of the super-resolution model using multiple metrics:
    1. PSNR (Peak Signal-to-Noise Ratio)
    2. SSIM (Structural Similarity Index Measure)
    3. MSE (Mean Squared Error)
    """
    model.eval()
    
    # Create dataloader for evaluation
    if num_samples and num_samples < len(dataset):
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        subset = torch.utils.data.Subset(dataset, indices)
        dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize metrics
    total_psnr = 0
    total_mse = 0
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)
    total_ssim = 0
    
    # Calculate PSNR function
    def calculate_psnr(sr, hr):
        # Ensure values are between [0, 1]
        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)
        
        # Calculate MSE
        mse = F.mse_loss(sr, hr)
        if mse == 0:
            return float('inf')
        
        # Calculate PSNR
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(dataloader, desc="Evaluating"):
            # Move to device
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            # Generate super-resolution images
            sr_imgs = model(lr_imgs)
            
            # Calculate PSNR for each image in batch
            for i in range(sr_imgs.size(0)):
                total_psnr += calculate_psnr(sr_imgs[i], hr_imgs[i])
            
            # Calculate MSE
            batch_mse = F.mse_loss(sr_imgs, hr_imgs).item()
            total_mse += batch_mse * lr_imgs.size(0)
            
            # Calculate SSIM
            batch_ssim = ssim_metric(sr_imgs, hr_imgs).item()
            total_ssim += batch_ssim * lr_imgs.size(0)
    
    # Calculate average metrics
    num_images = len(dataloader.dataset) if num_samples is None else min(num_samples, len(dataset))
    avg_psnr = total_psnr / num_images
    avg_mse = total_mse / num_images
    avg_ssim = total_ssim / len(dataloader)  # SSIM is calculated per batch
    
    print("\n===== Super-Resolution Model Accuracy =====")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {avg_psnr:.4f} dB")
    print(f"Structural Similarity Index (SSIM): {avg_ssim:.4f}")
    print(f"Mean Squared Error (MSE): {avg_mse:.6f}")
    print("==========================================")
    
    # Return metrics in case needed for further analysis
    return {
        'psnr': avg_psnr,
        'ssim': avg_ssim,
        'mse': avg_mse
    }

# ---- STEP 7: VISUALIZATION FUNCTION ----
def visualize_results(model_path, num_samples=3):
    """
    Visualize model results by comparing:
    1. Original low-resolution images
    2. Generated super-resolution images
    3. Original high-resolution images (ground truth)
    """
    # Load model
    model = EnhancedESRGAN().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Create dataset for visualization
    dataset = CustomDataset(HQ_FOLDER, LQ_FOLDER)
    
    # Randomly select samples
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    # Setup figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    
    # PSNR calculation function
    def calculate_psnr(sr, hr):
        sr = torch.clamp(sr, 0, 1)
        hr = torch.clamp(hr, 0, 1)
        mse = F.mse_loss(sr, hr)
        if mse == 0:
            return float('inf')
        max_pixel = 1.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        return psnr.item()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get image pair
            lr_img, hr_img = dataset[idx]
            
            # Generate super-resolution image
            lr_img_batch = lr_img.unsqueeze(0).to(device)
            sr_img = model(lr_img_batch).squeeze().cpu().clamp(0, 1)
            
            # Convert tensors to numpy arrays for plotting
            lr_img_np = lr_img.permute(1, 2, 0).numpy()
            sr_img_np = sr_img.permute(1, 2, 0).numpy()
            hr_img_np = hr_img.permute(1, 2, 0).numpy()
            
            # Plot images
            axes[i, 0].imshow(lr_img_np)
            axes[i, 0].set_title("Low Resolution")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(sr_img_np)
            axes[i, 1].set_title("Super Resolution (Generated)")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(hr_img_np)
            axes[i, 2].set_title("High Resolution (Original)")
            axes[i, 2].axis('off')
            
            # Calculate metrics for this sample
            psnr = calculate_psnr(sr_img.unsqueeze(0), hr_img.unsqueeze(0))
            ssim_metric = StructuralSimilarityIndexMeasure()
            ssim = ssim_metric(sr_img.unsqueeze(0), hr_img.unsqueeze(0)).item()
            
            fig.suptitle(f"Sample {i+1} - PSNR: {psnr:.2f}dB, SSIM: {ssim:.4f}", fontsize=16)
    
    plt.tight_layout()
    plt.savefig("ESRGAN_results_comparison.png")
    print("Results visualization saved as 'ESRGAN_results_comparison.png'")

# ---- STEP 8: MAIN CODE ----
if __name__ == "__main__":
    try:
        # Free up memory
        if device.type == 'mps':
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Start training with smaller batch size
        print("Starting training with reduced memory usage...")
        model = train_model(batch_size=4, num_epochs=20, lr=5e-4)
        
        # Evaluate model
        trainDataset = CustomDataset(HQ_FOLDER, LQ_FOLDER)
        print("Evaluating model performance...")
        evaluate_accuracy(model, trainDataset)
        
        # After training, visualize some results
        print("Visualizing model results...")
        visualize_results("ESRGAN_best.pth", num_samples=3)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()