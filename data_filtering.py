import os
import librosa
import numpy as np
import shutil
from tqdm import tqdm

# =========================
# Config
# =========================
SRC_DIR = r"D:\Tools\Projects\Diffusion-SVC\data\val\audio\norm"

DST_FULL = r"D:\Tools\Projects\Diffusion-SVC\data\val\audio\full"
DST_LOW  = r"D:\Tools\Projects\Diffusion-SVC\data\val\audio\low"

# Ngưỡng detect
FREQ_THRESHOLD = 20000   # Hz (48k: 16–18k | 24k: 10–11k)
ENERGY_RATIO   = 0.001    # ≥1% năng lượng nằm trên ngưỡng

N_FFT = 4096

# =========================
# Prepare
# =========================
os.makedirs(DST_FULL, exist_ok=True)
os.makedirs(DST_LOW, exist_ok=True)

full_count = 0
low_count = 0

# =========================
# Detect
# =========================
for fname in tqdm(os.listdir(SRC_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    path = os.path.join(SRC_DIR, fname)

    try:
        y, sr = librosa.load(path, sr=None, mono=True)

        S = np.abs(librosa.stft(y, n_fft=N_FFT))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

        total_energy = np.sum(S)
        high_energy = np.sum(S[freqs >= FREQ_THRESHOLD])

        ratio = high_energy / (total_energy + 1e-9)

        if ratio >= ENERGY_RATIO:
            shutil.copy2(path, os.path.join(DST_FULL, fname))
            full_count += 1
        else:
            if(low_count < 5):
                shutil.copy2(path, os.path.join(DST_LOW, fname))
                low_count += 1

    except Exception as e:
        print(f"Error processing {fname}: {e}")

# =========================
# Report
# =========================
print("Done.")
print(f"Full-range files : {full_count}")
print(f"Band-limited     : {low_count}")
