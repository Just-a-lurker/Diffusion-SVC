import os
import librosa
import numpy as np
import shutil
from tqdm import tqdm

# =========================
# Config
# =========================
SRC_DIR = r"D:\Tools\Projects\Diffusion-SVC\data\val\audio\norm"
DST_MALE = r"D:\Tools\Projects\Diffusion-SVC\data\val\audio\out"

os.makedirs(DST_MALE, exist_ok=True)

# Pitch config
FMIN = 60
FMAX = 300

QUANTILE = 0.25   # Q25
Q_LOW = 80        # Hz
Q_HIGH = 150      # Hz

# =========================
# Pitch util
# =========================
def pitch_quantile(y, sr):
    f0 = librosa.yin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr
    )
    f0 = f0[f0 > FMIN]
    if len(f0) < 10:
        return None
    return float(np.quantile(f0, QUANTILE))

# =========================
# Filter
# =========================
kept = 0
removed = 0

for fname in tqdm(os.listdir(SRC_DIR)):
    if not fname.lower().endswith(".wav"):
        continue

    path = os.path.join(SRC_DIR, fname)

    try:
        y, sr = librosa.load(path, sr=None, mono=True)
        y, _ = librosa.effects.trim(y, top_db=30)

        qf0 = pitch_quantile(y, sr)
        if qf0 is None:
            shutil.copy2(path, os.path.join(DST_OTHER, fname))
            removed += 1
            continue

        if Q_LOW <= qf0 <= Q_HIGH:
            shutil.copy2(path, os.path.join(DST_MALE, fname))
            kept += 1

    except Exception as e:
        print(f"Error {fname}: {e}")
        removed += 1

# =========================
# Report
# =========================
print("Done.")
print(f"Male kept : {kept}")
