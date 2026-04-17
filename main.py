import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models import inception_v3

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'

# -------------------------------
# DATASET PATH + SAVE PATH
# -------------------------------
dataset_path = "4"
save_path = "processed_images"
model_save_path = "trained_models"
eval_path = "evaluation_results"
os.makedirs(save_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(eval_path, exist_ok=True)

# -------------------------------
# PARAMETERS
# -------------------------------
MIN_WIDTH = 50
MIN_HEIGHT = 50
IMG_SIZE = 256
NORMALIZE_MODE = "0_1"
BATCH_SIZE = 8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🔧 Using device: {DEVICE}")

images = []
labels = []
names = []
metadata = []

# -------------------------------
# LOAD + CLEAN + RESIZE + NORMALIZE
# -------------------------------
print("📂 Loading dataset...")
for file in os.listdir(dataset_path):
    if file.endswith((".jpg", ".png")):
        image_path = os.path.join(dataset_path, file)
        label_path = os.path.join(dataset_path, file.rsplit('.', 1)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        try:
            img = Image.open(image_path).convert("RGB")
            if img.size[0] < MIN_WIDTH or img.size[1] < MIN_HEIGHT:
                continue

            img = img.resize((IMG_SIZE, IMG_SIZE))
            img_np = np.array(img).astype(np.float32)
            img_np /= 255.0 if NORMALIZE_MODE == "0_1" else (img_np / 127.5) - 1.0

            with open(label_path, "r") as f:
                text = f.read().strip()

            meta = {}
            for line in text.split("\n"):
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.lower().strip()] = v.strip()

            images.append(img_np)
            labels.append(text)
            metadata.append(meta)
            names.append(file)

        except Exception:
            continue

print(f"✅ Total clean images loaded: {len(images)}")

# -------------------------------
# FEATURE / META ENCODING
# -------------------------------
styles = sorted(set(m.get("style", "unknown") for m in metadata))
genres = sorted(set(m.get("genre", "unknown") for m in metadata))
artists = sorted(set(m.get("artist", "unknown") for m in metadata))

NUM_STYLES = len(styles)
NUM_GENRES = len(genres)
NUM_ARTISTS = len(artists)

print(f"📊 Dataset Statistics:")
print(f"   Styles: {NUM_STYLES}, Genres: {NUM_GENRES}, Artists: {NUM_ARTISTS}")


def one_hot(idx, size):
    v = np.zeros(size)
    v[idx] = 1
    return v


encoded_features = []
for m in metadata:
    style_id = styles.index(m.get("style", "unknown"))
    genre_id = genres.index(m.get("genre", "unknown"))
    artist_id = artists.index(m.get("artist", "unknown"))
    encoded_features.append({
        "style_id": style_id,
        "genre_id": genre_id,
        "artist_id": artist_id,
        "style_onehot": one_hot(style_id, NUM_STYLES),
        "genre_onehot": one_hot(genre_id, NUM_GENRES),
        "artist_onehot": one_hot(artist_id, NUM_ARTISTS)
    })


# -------------------------------
# CUSTOM DATASET CLASS
# -------------------------------
class ArtDataset(Dataset):
    def __init__(self, images, encoded_features):
        self.images = images
        self.features = encoded_features

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.FloatTensor(self.images[idx]).permute(2, 0, 1)  # HWC -> CHW
        style_vec = torch.FloatTensor(self.features[idx]["style_onehot"])
        return img, style_vec


# Split dataset
train_images, val_images, train_features, val_features = train_test_split(
    images, encoded_features, test_size=0.2, random_state=42
)

train_dataset = ArtDataset(train_images, train_features)
val_dataset = ArtDataset(val_images, val_features)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"✅ Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")


# ========================================
# MODEL ARCHITECTURES
# ========================================

# -------------------------------
# CONDITIONAL GAN GENERATOR
# -------------------------------
class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=NUM_STYLES):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        self.fc = nn.Linear(latent_dim + condition_dim, 256 * 8 * 8)

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 128x128
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 3, 4, 2, 1),  # 256x256
            nn.Tanh()
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(-1, 256, 8, 8)
        return self.conv_blocks(x)


# -------------------------------
# DISCRIMINATOR
# -------------------------------
class Discriminator(nn.Module):
    def __init__(self, condition_dim=NUM_STYLES):
        super(Discriminator, self).__init__()

        self.img_conv = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),  # 128x128
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(16, 32, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 8 * 8 + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, condition):
        img_features = self.img_conv(img)
        img_features = img_features.view(img_features.size(0), -1)
        x = torch.cat([img_features, condition], dim=1)
        return self.fc(x)


# -------------------------------
# PIX2PIX GENERATOR (U-Net)
# -------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super(UNetBlock, self).__init__()
        if down:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            if use_dropout:
                self.conv.add_module("dropout", nn.Dropout(0.5))

    def forward(self, x):
        return self.conv(x)


class Pix2PixGenerator(nn.Module):
    def __init__(self):
        super(Pix2PixGenerator, self).__init__()
        # Encoder
        self.down1 = nn.Conv2d(3, 64, 4, 2, 1)  # 128
        self.down2 = UNetBlock(64, 128, down=True)  # 64
        self.down3 = UNetBlock(128, 256, down=True)  # 32
        self.down4 = UNetBlock(256, 512, down=True)  # 16
        self.down5 = UNetBlock(512, 512, down=True)  # 8

        # Decoder
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)  # 16
        self.up2 = UNetBlock(1024, 256, down=False)  # 32
        self.up3 = UNetBlock(512, 128, down=False)  # 64
        self.up4 = UNetBlock(256, 64, down=False)  # 128
        self.up5 = nn.ConvTranspose2d(128, 3, 4, 2, 1)  # 256
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))
        u5 = self.up5(torch.cat([u4, d1], dim=1))

        return self.tanh(u5)


# ========================================
# TRAINING FUNCTIONS
# ========================================

def train_cgan(generator, discriminator, train_loader, num_epochs=NUM_EPOCHS):
    """Train Conditional GAN"""
    print("\n🚀 Training Conditional GAN...")

    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        for real_imgs, conditions in train_loader:
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            conditions = conditions.to(DEVICE)

            # Normalize images to [-1, 1] for tanh output
            real_imgs = (real_imgs - 0.5) * 2

            # Labels
            real_labels = torch.ones(batch_size, 1).to(DEVICE)
            fake_labels = torch.zeros(batch_size, 1).to(DEVICE)

            # Train Discriminator
            d_optimizer.zero_grad()

            real_output = discriminator(real_imgs, conditions)
            d_real_loss = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, generator.latent_dim).to(DEVICE)
            fake_imgs = generator(noise, conditions)
            fake_output = discriminator(fake_imgs.detach(), conditions)
            d_fake_loss = criterion(fake_output, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            g_optimizer.zero_grad()

            fake_output = discriminator(fake_imgs, conditions)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()

            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        avg_g_loss = epoch_g_loss / len(train_loader)
        avg_d_loss = epoch_d_loss / len(train_loader)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] | G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")

    return g_losses, d_losses


# ========================================
# EVALUATION METRICS
# ========================================

def calculate_fid(real_images, generated_images, batch_size=50):
    """Calculate Fréchet Inception Distance"""
    print("\n📊 Calculating FID...")

    inception_model = inception_v3(pretrained=True, transform_input=False).to(DEVICE)
    inception_model.fc = nn.Identity()
    inception_model.eval()

    def get_features(images):
        features = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_tensor = torch.FloatTensor(batch).permute(0, 3, 1, 2).to(DEVICE)
            batch_tensor = torch.nn.functional.interpolate(batch_tensor, size=(299, 299), mode='bilinear')

            with torch.no_grad():
                feat = inception_model(batch_tensor)
            features.append(feat.cpu().numpy())
        return np.concatenate(features, axis=0)

    real_features = get_features(real_images[:min(500, len(real_images))])
    gen_features = get_features(generated_images[:min(500, len(generated_images))])

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index"""
    from skimage.metrics import structural_similarity
    return structural_similarity(img1, img2, multichannel=True, channel_axis=2)


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# ========================================
# INTERACTIVE VISUALIZATION
# ========================================

def create_style_comparison_dashboard(generator, styles, save_dir=eval_path):
    """Create interactive style switching visualization"""
    print("\n🎨 Creating style comparison dashboard...")

    generator.eval()
    num_samples = 5

    fig, axes = plt.subplots(num_samples, len(styles), figsize=(4 * len(styles), 4 * num_samples))

    for i in range(num_samples):
        noise = torch.randn(1, generator.latent_dim).to(DEVICE)

        for j, style in enumerate(styles):
            condition = torch.zeros(1, NUM_STYLES).to(DEVICE)
            condition[0, j] = 1

            with torch.no_grad():
                generated = generator(noise, condition)

            img = generated.squeeze().cpu().permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            img = np.clip(img, 0, 1)

            ax = axes[i, j] if num_samples > 1 else axes[j]
            ax.imshow(img)
            if i == 0:
                ax.set_title(f"{style}", fontweight='bold')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'style_comparison_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Dashboard saved to {save_dir}/style_comparison_dashboard.png")


# ========================================
# MAIN EXECUTION
# ========================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🎨 ADVANCED GAN TRAINING & EVALUATION PIPELINE")
    print("=" * 60)

    # Initialize models
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    print(f"\n🔧 Model Parameters:")
    print(f"   Generator params: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"   Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Train cGAN
    g_losses, d_losses = train_cgan(generator, discriminator, train_loader, num_epochs=NUM_EPOCHS)

    # Save models
    torch.save(generator.state_dict(), os.path.join(model_save_path, 'cgan_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(model_save_path, 'cgan_discriminator.pth'))
    print(f"\n💾 Models saved to {model_save_path}")

    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label='Generator Loss', linewidth=2)
    plt.plot(d_losses, label='Discriminator Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Generate samples for evaluation
    print("\n🖼️  Generating samples for evaluation...")
    generator.eval()
    generated_images = []

    for i in range(min(100, len(val_images))):
        noise = torch.randn(1, generator.latent_dim).to(DEVICE)
        style_idx = encoded_features[i]["style_id"]
        condition = torch.zeros(1, NUM_STYLES).to(DEVICE)
        condition[0, style_idx] = 1

        with torch.no_grad():
            gen_img = generator(noise, condition)

        img = gen_img.squeeze().cpu().permute(1, 2, 0).numpy()
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        generated_images.append(img)

    # Calculate metrics
    real_subset = val_images[:len(generated_images)]

    ssim_scores = [calculate_ssim(real_subset[i], generated_images[i])
                   for i in range(len(generated_images))]
    psnr_scores = [calculate_psnr(real_subset[i], generated_images[i])
                   for i in range(len(generated_images))]

    avg_ssim = np.mean(ssim_scores)
    avg_psnr = np.mean(psnr_scores)

    print(f"\n📈 Evaluation Metrics:")
    print(f"   Average SSIM: {avg_ssim:.4f}")
    print(f"   Average PSNR: {avg_psnr:.2f} dB")

    # Plot metrics
    plt.subplot(1, 2, 2)
    metrics = ['SSIM', 'PSNR/10']
    values = [avg_ssim, avg_psnr / 10]
    bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c'], width=0.6)
    plt.ylabel('Score')
    plt.title('Image Quality Metrics')
    plt.ylim(0, 1)

    for bar, val in zip(bars, [avg_ssim, avg_psnr]):
        height = bar.get_height()
        label = f'{val:.3f}' if val < 10 else f'{val:.1f}'
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 label, ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(eval_path, 'training_metrics.png'), dpi=150)
    plt.show()

    # Create style comparison dashboard
    create_style_comparison_dashboard(generator, styles[:min(5, len(styles))])

    # Display generated samples
    print("\n🎨 Displaying generated samples...")
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i in range(min(10, len(generated_images))):
        row = i // 5
        col = i % 5
        axes[row, col].imshow(generated_images[i])
        axes[row, col].set_title(f"Sample {i + 1}")
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(eval_path, 'generated_samples.png'), dpi=150)
    plt.show()

    print("\n" + "=" * 60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n📁 Results saved in:")
    print(f"   - Models: {model_save_path}")
    print(f"   - Evaluations: {eval_path}")
    print(f"   - Processed images: {save_path}")