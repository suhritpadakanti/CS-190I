import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from model import Discriminator, Generator, initialize_weights
import os

# --- Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE_GEN = 1e-4
LEARNING_RATE_DISC = 4e-4  # For SAGAN, it's common to have a faster D learning rate
BATCH_SIZE = 32
IMAGE_SIZE = 64
CHANNELS_IMG = 3
NOISE_DIM = 100
NUM_EPOCHS = 5000
FEATURES_GEN = 64
FEATURES_DISC = 64

# --- Data Loading and On-the-Fly Augmentation ---
data_transforms = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(CHANNELS_IMG)], [0.5 for _ in range(CHANNELS_IMG)]),
])

dataset = datasets.ImageFolder(root="./data", transform=data_transforms)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# --- Main Training Function ---
def train():
    os.makedirs("generated_images", exist_ok=True)
    
    gen = Generator(NOISE_DIM, CHANNELS_IMG, FEATURES_GEN).to(device)
    disc = Discriminator(CHANNELS_IMG, FEATURES_DISC).to(device)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.0, 0.9))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.0, 0.9))
    criterion = nn.BCELoss()

    fixed_noise = torch.randn(32, NOISE_DIM, 1, 1).to(device)
    
    gen.train()
    disc.train()

    print("🚀 Starting Training with Self-Attention GAN...")
    for epoch in range(NUM_EPOCHS):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real = real_images.to(device)
            noise = torch.randn(real.size(0), NOISE_DIM, 1, 1).to(device)
            fake = gen(noise)

            # --- Train Discriminator ---
            disc.zero_grad()
            disc_real = disc(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real)) # Use real labels for hinge loss variation
            disc_fake = disc(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake)
            loss_disc.backward()
            opt_disc.step()

            # --- Train Generator ---
            gen.zero_grad()
            output = disc(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            loss_gen.backward()
            opt_gen.step()
            
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} | "
                    f"Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}"
                )
                with torch.no_grad():
                    fake_samples = gen(fixed_noise)
                    vutils.save_image(
                        fake_samples.detach(),
                        f"generated_images/sagan_epoch_{epoch}.png",
                        normalize=True,
                        nrow=8
                    )

if __name__ == "__main__":
    train()