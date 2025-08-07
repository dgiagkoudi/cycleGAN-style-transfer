# cyclegan_kfold_train.py

import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

# ==== CycleGAN Components ===
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        ]

        for _ in range(9):
            model.append(ResidualBlock(256))

        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# === Dataset Loading ===
class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(self, rootA, rootB, transform):
        self.dsA = datasets.ImageFolder(root=rootA, transform=transform)
        self.dsB = datasets.ImageFolder(root=rootB, transform=transform)

    def __len__(self):
        return min(len(self.dsA), len(self.dsB))

    def __getitem__(self, index):
        return self.dsA[index][0], self.dsB[index][0]

# === Training Setup ===
def train_kfold(rootA, rootB, device, epochs=8, k=5):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = UnpairedDataset(rootA, rootB, transform)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"\nFold {fold + 1}/{k}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=1, shuffle=False)

        G_AB = Generator(3, 3).to(device)
        D_B = Discriminator(3).to(device)

        g_optimizer = optim.Adam(G_AB.parameters(), lr=2e-4, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(D_B.parameters(), lr=2e-4, betas=(0.5, 0.999))

        criterion_gan = nn.MSELoss()
        criterion_cycle = nn.L1Loss()

        for epoch in range(epochs):
            g_losses = []
            d_losses = []
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for real_A, real_B in loop:
                real_A, real_B = real_A.to(device), real_B.to(device)
                valid = torch.ones((real_B.size(0), 1, 30, 30), device=device)
                fake = torch.zeros_like(valid)

                # === Train Generator ===
                G_AB.train()
                fake_B = G_AB(real_A)
                pred_fake = D_B(fake_B)
                loss_GAN = criterion_gan(pred_fake, valid)
                loss_cycle = criterion_cycle(G_AB(fake_B), real_A)
                g_loss = loss_GAN + 10 * loss_cycle

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

                # === Train Discriminator ===
                pred_real = D_B(real_B)
                pred_fake = D_B(fake_B.detach())

                loss_real = criterion_gan(pred_real, valid)
                loss_fake = criterion_gan(pred_fake, fake)
                d_loss = 0.5 * (loss_real + loss_fake)

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                loop.set_postfix({"g_loss": g_loss.item(), "d_loss": d_loss.item()})
                g_losses.append(g_loss.item())
                d_losses.append(d_loss.item())

            avg_g_loss = np.mean(g_losses)
            avg_d_loss = np.mean(d_losses)
            print(f"Fold {fold+1}, Epoch {epoch+1}: Avg G Loss = {avg_g_loss:.4f}, Avg D Loss = {avg_d_loss:.4f}")

            # === Save losses to file ===
            np.savetxt(f"fold{fold+1}_epoch{epoch+1}_gen_loss.txt", g_losses)
            np.savetxt(f"fold{fold+1}_epoch{epoch+1}_disc_loss.txt", d_losses)

            # === Save model ===
            torch.save(G_AB.state_dict(), f"G_AB_fold{fold+1}_epoch{epoch+1}.pt")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Χρήση συσκευής: {device}")
    train_kfold("./datamonet/trainA", "./datamonet/trainB", device)