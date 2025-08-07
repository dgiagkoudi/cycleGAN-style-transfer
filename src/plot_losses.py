# plot_losses.py

import os
import numpy as np
import matplotlib.pyplot as plt

# === Ρυθμίσεις ===
folds = [1, 2]  # Τροποποίησε αν έχεις παραπάνω
epochs = 8

# === Διαδρομή αρχείων ===
base_path = "./"  

# === Δημιουργία γραφημάτων ===
for fold in folds:
    gen_means = []
    disc_means = []

    for epoch in range(1, epochs + 1):
        gen_path = os.path.join(base_path, f"fold{fold}_epoch{epoch}_gen_loss.txt")
        disc_path = os.path.join(base_path, f"fold{fold}_epoch{epoch}_disc_loss.txt")

        if os.path.exists(gen_path) and os.path.exists(disc_path):
            gen_loss = np.loadtxt(gen_path)
            disc_loss = np.loadtxt(disc_path)

            gen_means.append(np.mean(gen_loss))
            disc_means.append(np.mean(disc_loss))
        else:
            print(f"Λείπει αρχείο για fold {fold}, epoch {epoch}")
            gen_means.append(np.nan)
            disc_means.append(np.nan)

    # === Σχεδίαση ===
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), gen_means, label="Generator Loss", marker='o')
    plt.plot(range(1, epochs + 1), disc_means, label="Discriminator Loss", marker='x')
    plt.title(f"Fold {fold} - Loss ανά Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Μέσο Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # === Αποθήκευση ===
    plt.savefig(f"loss_curve_fold{fold}.png")
    print(f"Αποθηκεύτηκε: loss_curve_fold{fold}.png")
    plt.close()

print("Ολοκληρώθηκε η παραγωγή διαγραμμάτων!")