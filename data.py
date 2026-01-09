from datasets import load_dataset #36
import soundfile as sf
import os
import random

# Load toàn bộ dataset (download hết)
ds = load_dataset(
    "doof-ferb/infore1_25hours",
    split="train"
)

train_dir = "data/train/audio/1"
val_dir   = "data/val/audio/1"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Tổng số sample
num_samples = len(ds)

# Chọn ngẫu nhiên 10 index làm validation
val_indices = set(random.sample(range(num_samples), 10))

train_idx = 0
val_idx = 0

for i, sample in enumerate(ds):
    audio = sample["audio"]
    wav = audio["array"]
    sr = audio["sampling_rate"]

    if i in val_indices:
        out_path = f"{val_dir}/{val_idx:04d}.wav"
        val_idx += 1
    else:
        out_path = f"{train_dir}/{train_idx:04d}.wav"
        train_idx += 1

    sf.write(out_path, wav, sr)

print(f"Done. Train: {train_idx}, Val: {val_idx}")
