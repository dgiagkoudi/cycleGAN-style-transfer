import os
import random
import time
import itertools
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Ρύθμιση συσκευής ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f" Χρήση συσκευής: {device}")

# === Μετασχηματισμοί εικόνας ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
])

# === Dataset για μη συζευγμένες εικόνες ===
class UnpairedDataset(Dataset):
    def __init__(self, pathA, pathB, transform=None):
        self.files_A = list(Path(pathA).glob("*"))
        self.files_B = list(Path(pathB).glob("*"))
        self.transform = transform

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("RGB")
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return {"A": img_A, "B": img_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# === Αρχιτεκτονική Residual Block ===
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)

# === Generator ===
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        # Downsampling
        in_feat = 64
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, in_feat * 2, 3, 2, 1),
                nn.InstanceNorm2d(in_feat * 2),
                nn.ReLU(True)
            ]
            in_feat *= 2
        # Residual Blocks
        for _ in range(9):
            model += [ResidualBlock(in_feat)]
        # Upsampling
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_feat, in_feat // 2, 3, 2, 1, output_padding=1),
                nn.InstanceNorm2d(in_feat // 2),
                nn.ReLU(True)
            ]
            in_feat //= 2
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, 7),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# === Discriminator ===
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, True)
        ]
        in_feat = 64
        for _ in range(3):
            model += [
                nn.Conv2d(in_feat, in_feat * 2, 4, 2, 1),
                nn.InstanceNorm2d(in_feat * 2),
                nn.LeakyReLU(0.2, True)
            ]
            in_feat *= 2
        model += [
            nn.Conv2d(in_feat, 1, 4, padding=1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# === Βοηθητική συνάρτηση ===
def get_target(pred, real=True):
    return torch.ones_like(pred) if real else torch.zeros_like(pred)

# === Ρυθμίσεις ===
dataset_root = "datamonet"  # φάκελος με trainA/trainB/testA/testB
batch_size = 1
epochs = 20
lr = 0.0002
lambda_cyc = 10.0
lambda_idt = 5.0

# === Φόρτωση δεδομένων ===
dataloader = DataLoader(
    UnpairedDataset(
        os.path.join(dataset_root, "trainA"),
        os.path.join(dataset_root, "trainB"),
        transform
    ),
    batch_size=batch_size,
    shuffle=True
)

# === Μοντέλα ===
G_AB = Generator(3, 3).to(device)
G_BA = Generator(3, 3).to(device)
D_A = Discriminator(3).to(device)
D_B = Discriminator(3).to(device)

# === Απώλειες και Βελτιστοποιητές ===
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_idt = nn.L1Loss()

optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=lr, betas=(0.5, 0.999))

# === Εκπαίδευση ===
for epoch in range(1, epochs + 1):
    print(f"\n Epoch {epoch}/{epochs}")
    for i, data in enumerate(tqdm(dataloader)):
        real_A = data["A"].to(device)
        real_B = data["B"].to(device)

        # === Train Generators ===
        optimizer_G.zero_grad()

        fake_B = G_AB(real_A)
        fake_A = G_BA(real_B)

        loss_GAN_AB = criterion_GAN(D_B(fake_B), get_target(D_B(fake_B), True).to(device))
        loss_GAN_BA = criterion_GAN(D_A(fake_A), get_target(D_A(fake_A), True).to(device))

        reconstr_A = G_BA(fake_B)
        reconstr_B = G_AB(fake_A)
        loss_cycle_A = criterion_cycle(reconstr_A, real_A)
        loss_cycle_B = criterion_cycle(reconstr_B, real_B)

        idt_A = G_BA(real_A)
        idt_B = G_AB(real_B)
        loss_idt_A = criterion_idt(idt_A, real_A)
        loss_idt_B = criterion_idt(idt_B, real_B)

        loss_G = (loss_GAN_AB + loss_GAN_BA +
                  lambda_cyc * (loss_cycle_A + loss_cycle_B) +
                  lambda_idt * (loss_idt_A + loss_idt_B))

        loss_G.backward()
        optimizer_G.step()

        # === Train Discriminator A ===
        optimizer_D_A.zero_grad()
        loss_D_real_A = criterion_GAN(D_A(real_A), get_target(D_A(real_A), True).to(device))
        loss_D_fake_A = criterion_GAN(D_A(fake_A.detach()), get_target(D_A(fake_A.detach()), False).to(device))
        loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
        loss_D_A.backward()
        optimizer_D_A.step()

        # === Train Discriminator B ===
        optimizer_D_B.zero_grad()
        loss_D_real_B = criterion_GAN(D_B(real_B), get_target(D_B(real_B), True).to(device))
        loss_D_fake_B = criterion_GAN(D_B(fake_B.detach()), get_target(D_B(fake_B.detach()), False).to(device))
        loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
        loss_D_B.backward()
        optimizer_D_B.step()

    # === Αποθήκευση μοντέλων ===
    torch.save(G_AB.state_dict(), f"G_AB_epoch{epoch}.pt")
    torch.save(G_BA.state_dict(), f"G_BA_epoch{epoch}.pt")
    print(f"Αποθηκεύτηκαν τα μοντέλα εποχής {epoch}")