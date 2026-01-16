from datasets import load_dataset #36
import soundfile as sf
from datasets import Audio
import os
import random

# Load toàn bộ dataset (download hết)
DATASET_NAME = "doof-ferb/infore1_25hours"
TRAIN_DIR = "data/train/audio/1"
VAL_DIR   = "data/val/audio/1"
VAL_SIZE = 10
SEED = 42

random.seed(SEED)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
#
print("Loading full dataset...")
ds = load_dataset(DATASET_NAME, split="train")  # tải hẳn dataset về
print(f"Total samples in dataset: {len(ds)}")

num_samples = len(ds)
val_indices = set(random.sample(range(num_samples), VAL_SIZE))

train_idx = 0
val_idx = 0

for i, sample in enumerate(ds):
    audio = sample["audio"]
    wav = audio["array"]
    sr = audio["sampling_rate"]

    if i in val_indices:
        out_path = os.path.join(VAL_DIR, f"{val_idx:04d}.wav")
        val_idx += 1
    else:
        out_path = os.path.join(TRAIN_DIR, f"{train_idx:04d}.wav")
        train_idx += 1

    sf.write(out_path, wav, sr)

print(f"Done. Train: {train_idx}, Val: {val_idx}")

# DATASET_NAME = "pnnbao-ump/VieNeu-TTS-140h"
# TRAIN_DIR = "data/train/audio/2"
# VAL_DIR   = "data/val/audio/2"
# VAL_SIZE  = 10
# SEED = 42
#
# random.seed(SEED)
# os.makedirs(TRAIN_DIR, exist_ok=True)
# os.makedirs(VAL_DIR, exist_ok=True)
#
# print("Loading full dataset...")
# ds = load_dataset(DATASET_NAME, split="train")  # tải hẳn dataset về
# print(f"Total samples in dataset: {len(ds)}")
#
# def filter_fn(x):
#     return x["gender"] == "male"
#
# ds = ds.filter(filter_fn)
# print(f"Samples after filter: {len(ds)}")
# ds = ds.cast_column("audio", Audio(sampling_rate=24000))
# num_samples = len(ds)
# val_indices = set(random.sample(range(num_samples), VAL_SIZE))
#
# train_idx = 0
# val_idx = 0
#
# for i, sample in enumerate(ds):
#     audio = sample['audio']
#     wav = sample['audio']['array']
#     sr = audio['sampling_rate']
#
#     if i in val_indices:
#         out_path = os.path.join(VAL_DIR, f"{val_idx:04d}.wav")
#         val_idx += 1
#     else:
#         out_path = os.path.join(TRAIN_DIR, f"{train_idx:04d}.wav")
#         train_idx += 1
#
#     sf.write(out_path, wav, sr)
#
# print(f"Done. Train: {train_idx}, Val: {val_idx}")