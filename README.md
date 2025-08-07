# Μεταφορά Στυλ με CycleGAN (Monet ➝ Φωτογραφία)

Το έργο αυτό υλοποιεί μεταφορά στυλ (Style Transfer) με χρήση **Generative Adversarial Networks (GANs)** και συγκεκριμένα την αρχιτεκτονική **CycleGAN**, για τη μετατροπή ζωγραφιών του Claude Monet σε ρεαλιστικές φωτογραφικές εικόνες.

---

## Βασικά Χαρακτηριστικά

- Εκπαίδευση **CycleGAN** με χρήση 9 Residual Blocks
- Μεταφορά στυλ **χωρίς αντιστοιχισμένα ζεύγη εικόνων** (Unpaired Image Translation)
- Υλοποίηση **5-Fold Cross-Validation**
- Πειραματική αξιολόγηση με τις μετρικές **PSNR** και **SSIM**
- Στατιστική ανάλυση με **T-Test** για τις μεταβολές ανά epoch

---

## Εκτέλεση Κώδικα

1. Εκπαίδευση CycleGAN με k-fold cross-validation:
python src/cyclegan_kfold_train.py
2. Οπτική σύγκριση μεταξύ Epoch 1 και Epoch 8:
python src/compare_epochs.py
3. Δημιουργία καμπυλών απωλειών (loss curves):
python src/plot_losses.py
4. Υπολογισμός μετρικών ποιότητας (PSNR / SSIM):
python src/compare_metrics_from_outputs.py
5. Στατιστική αξιολόγηση (t-test):
python src/test.py

---

## Παραγόμενα Αποτελέσματα

Epoch Comparison: Οπτικές συγκρίσεις παραγόμενων εικόνων από διαφορετικά στάδια εκπαίδευσης

Loss Curves: Καμπύλες απωλειών Generator / Discriminator για κάθε fold

Μετρήσεις PSNR & SSIM: Πίνακες και διαγράμματα αξιολόγησης ποιότητας εικόνας

Στατιστικός Έλεγχος: Αποτελέσματα t-test για στατιστικά σημαντικές βελτιώσεις

---

## Dataset

Το dataset Monet2Photo χρησιμοποιήθηκε για την εκπαίδευση του CycleGAN και προέρχεται από δημόσια πηγή. Δεν περιλαμβάνεται στο αποθετήριο λόγω περιορισμών μεγέθους και πνευματικών δικαιωμάτων.

---
## Εκπαιδευμένα Μοντέλα

Τα εκπαιδευμένα μοντέλα `.pt` δεν περιλαμβάνονται στο αποθετήριο λόγω περιορισμών μεγέθους αρχείων στο GitHub.

---

## Δομή Φακέλων

```
CycleGAN_Project/
│
├── src/                         # Αρχεία κώδικα
│   ├── cyclegan_train.py
│   ├── cyclegan_kfold_train.py
│   ├── compare_epochs.py
│   ├── plot_losses.py
│   ├── compare_metrics_from_outputs.py
│   └── test.py
│
├── outputs/                     # Παραγόμενες εικόνες και διαγράμματα
│   ├── outputs_compare/
│   ├── loss_curve_fold1.png
│   └── loss_curve_fold2.png
│
├── report/                      # Τελική αναφορά (PDF)
│   └── reportGAN.pdf
│
├── loss_logs/
│   ├── fold1_epoch1_gen_loss.txt
│   ├── fold1_epoch1_disc_loss.txt
│   ├── fold1_epoch2_gen_loss.txt
│
├── requirements.txt             # Βιβλιοθήκες που απαιτούνται
