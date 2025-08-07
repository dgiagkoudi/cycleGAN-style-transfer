# compare_metrics_from_outputs.py

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# === Φάκελοι ===
input_folder = Path("datamonet/testA/class_dummy")
output_folder = Path("outputs_compare")

# === Φόρτωση μετασχηματισμών ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

inv_transform = transforms.Compose([
    transforms.Normalize(mean=[0, 0, 0], std=[1/0.5]*3),
    transforms.Normalize(mean=[-0.5]*3, std=[1]*3)
])

# === Προετοιμασία ===
input_paths = sorted(input_folder.glob("*.jpg"))[:5]

metrics = []

for i in range(5):
    input_img = Image.open(input_paths[i]).convert("RGB")
    fake1_img = Image.open(output_folder / f"fake_B_epoch1_img{i}.png").convert("RGB")
    fake8_img = Image.open(output_folder / f"fake_B_epoch8_img{i}.png").convert("RGB")

    input_np = np.array(input_img.resize((256, 256))) / 255.0
    fake1_np = np.array(fake1_img.resize((256, 256))) / 255.0
    fake8_np = np.array(fake8_img.resize((256, 256))) / 255.0

    psnr1 = psnr(input_np, fake1_np, data_range=1.0)
    ssim1 = ssim(input_np, fake1_np, channel_axis=-1, data_range=1.0)

    psnr8 = psnr(input_np, fake8_np, data_range=1.0)
    ssim8 = ssim(input_np, fake8_np, channel_axis=-1, data_range=1.0)

    metrics.append([psnr1, ssim1, psnr8, ssim8])
    print(f"Εικόνα {i+1} - Epoch1: PSNR={psnr1:.2f}, SSIM={ssim1:.3f} | Epoch8: PSNR={psnr8:.2f}, SSIM={ssim8:.3f}")

# === Τελικός πίνακας ===
print("\nΠίνακας 1: Ποσοτικές Μετρικές PSNR / SSIM ανά Epoch")
print("{:<10} {:<10} {:<10} {:<10} {:<10}".format("Image", "PSNR_1", "SSIM_1", "PSNR_8", "SSIM_8"))
for i, row in enumerate(metrics):
    print("{:<10} {:<10.2f} {:<10.3f} {:<10.2f} {:<10.3f}".format(f"Img{i+1}", *row))