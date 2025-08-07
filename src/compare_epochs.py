import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
import os
from matplotlib import pyplot as plt
import torch.nn as nn

# === Generator ===
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

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        ]
        in_feat = 64
        for _ in range(2):
            model += [
                nn.Conv2d(in_feat, in_feat * 2, 3, 2, 1),
                nn.InstanceNorm2d(in_feat * 2),
                nn.ReLU(True)
            ]
            in_feat *= 2
        for _ in range(9):
            model += [ResidualBlock(in_feat)]
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

# === Συσκευή ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Χρήση συσκευής: {device}")
print("Τρέχει το compare_epochs.py ")

# === Φόρτωση των 2 μοντέλων ===
generator_epoch1 = Generator(3, 3).to(device)
generator_epoch1.load_state_dict(torch.load("G_AB_epoch1.pt", map_location=device))
generator_epoch1.eval()

generator_epoch8 = Generator(3, 3).to(device)
generator_epoch8.load_state_dict(torch.load("G_AB_epoch8.pt", map_location=device))
generator_epoch8.eval()

print("Φορτώθηκαν τα μοντέλα από Epoch 1 και 8")

# === Ρυθμίσεις ===
input_folder = "datamonet/testA"
output_folder = "outputs_compare"
Path(output_folder).mkdir(exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Δοκιμή σε 5 εικόνες ===
input_paths = list(Path(input_folder).glob("*.jpg"))[:5]

for i, img_path in enumerate(input_paths):
    img = Image.open(img_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        fake_1 = generator_epoch1(input_tensor)
        fake_8 = generator_epoch8(input_tensor)

    fake_1 = (fake_1 + 1) / 2.0
    fake_8 = (fake_8 + 1) / 2.0

    save_image(fake_1, f"{output_folder}/fake_B_epoch1_img{i}.png")
    save_image(fake_8, f"{output_folder}/fake_B_epoch8_img{i}.png")
    print(f"Εικόνα {i} αποθηκεύτηκε για Epoch 1 & 8")

print("Ολοκλήρωση σύγκρισης Epoch 1 vs Epoch 8")