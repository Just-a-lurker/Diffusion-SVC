import os
import torch
import numpy as np
import pandas as pd
from speechbrain.inference.speaker import EncoderClassifier
#

# =========================
# CONFIG (EDIT THESE)
# =========================

SPEAKER_NAME = "Female"

REFERENCE_DIR = f"E:\Projects\Diffusion-SVC\data/train/audio/1"
CONVERTED_DIR = f"data/test/SpeakerSims/Converted/{SPEAKER_NAME}"

# =========================
# LOAD MODEL
# =========================

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/ecapa",
    run_opts={"device": "cuda:0"}
)

# =========================
# FUNCTIONS
# =========================

import os
import torchaudio
import soundfile as sf
import torch
def load_embeddings_from_dir(directory):
    embeddings = []

    for file in os.listdir(directory):
        if file.endswith(".wav"):
            path = os.path.join(directory, file)

            audio, fs = sf.read(path)

            # convert to torch tensor
            signal = torch.tensor(audio).float()

            # if stereo → convert to mono
            if len(signal.shape) > 1:
                signal = signal.mean(dim=1)

            # add batch dimension for SpeechBrain
            signal = signal.unsqueeze(0)

            emb = classifier.encode_batch(signal).squeeze(1)
            embeddings.append(emb)

    return embeddings


def average_embedding(embeddings):
    """Average a list of speaker embeddings."""
    return torch.stack(embeddings).mean(dim=0)


def cosine_similarity(a, b):
    """Compute cosine similarity between two embeddings."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).item()

# =========================
# LOAD REFERENCE SPEAKER EMBEDDING
# =========================

ref_embeddings = load_embeddings_from_dir(REFERENCE_DIR)

if len(ref_embeddings) == 0:
    raise ValueError(f"No reference audio found in {REFERENCE_DIR}")

ref_embedding = average_embedding(ref_embeddings)

print(f"Loaded {len(ref_embeddings)} reference files for {SPEAKER_NAME}")

# =========================
# EVALUATE CONVERTED FILES
# =========================

results = []

conv_embeddings = load_embeddings_from_dir(CONVERTED_DIR)

if len(conv_embeddings) == 0:
    raise ValueError(f"No converted audio found in {CONVERTED_DIR}")

for i, emb in enumerate(conv_embeddings):

    sim = cosine_similarity(ref_embedding, emb)

    results.append({
        "sample_id": i,
        "speaker": SPEAKER_NAME,
        "similarity": sim
    })

# =========================
# RESULTS
# =========================

df = pd.DataFrame(results)

print("\nPer-sample results:")
print(df)

print("\nSummary:")
print(f"Speaker: {SPEAKER_NAME}")
print(f"Mean similarity: {df['similarity'].mean():.4f}")
print(f"Std deviation: {df['similarity'].std():.4f}")
print(f"Min: {df['similarity'].min():.4f}")
print(f"Max: {df['similarity'].max():.4f}")