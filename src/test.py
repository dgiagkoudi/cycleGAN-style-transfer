from scipy.stats import ttest_rel
import numpy as np

# PSNR και SSIM τιμές από Epoch 1 και Epoch 8
psnr_1 = np.array([11.41, 12.82, 13.27, 13.53, 15.54])
psnr_8 = np.array([16.18, 11.37, 14.04, 15.57, 14.75])

ssim_1 = np.array([0.361, 0.415, 0.426, 0.361, 0.325])
ssim_8 = np.array([0.537, 0.551, 0.633, 0.556, 0.443])

# Υπολογισμός μέσης τιμής και διασποράς
print("Μέσες τιμές & τυπικές αποκλίσεις:")
print(f"PSNR Epoch 1: {np.mean(psnr_1):.2f} ± {np.std(psnr_1):.2f}")
print(f"PSNR Epoch 8: {np.mean(psnr_8):.2f} ± {np.std(psnr_8):.2f}")
print(f"SSIM Epoch 1: {np.mean(ssim_1):.3f} ± {np.std(ssim_1):.3f}")
print(f"SSIM Epoch 8: {np.mean(ssim_8):.3f} ± {np.std(ssim_8):.3f}")

# Paired t-test μεταξύ των δύο epochs
t_psnr, p_psnr = ttest_rel(psnr_8, psnr_1)
t_ssim, p_ssim = ttest_rel(ssim_8, ssim_1)

print("\nT-Test Αποτελέσματα:")
print(f"PSNR: t = {t_psnr:.3f}, p = {p_psnr:.4f}")
print(f"SSIM: t = {t_ssim:.3f}, p = {p_ssim:.4f}")

# Συμπερασματικό μήνυμα
if p_psnr < 0.05 and p_ssim < 0.05:
    print("\nΗ διαφορά μεταξύ Epoch 1 και 8 είναι στατιστικά σημαντική (p < 0.05).")
else:
    print("\nΔεν υπάρχει στατιστικά σημαντική διαφορά (p ≥ 0.05).")