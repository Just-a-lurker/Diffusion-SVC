import os
import soundfile as sf
from datasets import load_dataset
#pip install openai-whisper jiwer
#https://www.gyan.dev/ffmpeg/builds/#release-builds
# ====== CONFIG ======
DATASET_NAME = "dolly-vn/dolly-audio-1000h-vietnamese"
SPLIT = "train"
OUTPUT_DIR = "test"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# load dataset
ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)

subset = ds.skip(50).take(50)

transcript_path = os.path.join(OUTPUT_DIR, "transcripts.txt")

with open(transcript_path, "w", encoding="utf-8") as f:
    START_IDX = 50

    for i, row in enumerate(subset, start=START_IDX):

        text = row.get("text", "")

        audio = row.get("audio", None)

        audio_file_path = None

        # ====== CASE 1: audio là dict (HF Audio feature) ======
        if isinstance(audio, dict):
            if audio.get("bytes"):
                print("HELLO")
                audio_file_path = os.path.join(OUTPUT_DIR, f"{i}.wav")

                with open(audio_file_path, "wb") as f_audio:
                    f_audio.write(audio["bytes"])

            elif "array" in audio and audio["array"] is not None:
                # decode sẵn trong RAM -> save wav
                audio_file_path = os.path.join(OUTPUT_DIR, f"{i}.wav")
                sf.write(audio_file_path, audio["array"], audio.get("sampling_rate", 16000))

            elif "path" in audio and audio["path"]:
                # file gốc (đã cache hoặc local)
                src_path = audio["path"]
                ext = os.path.splitext(src_path)[-1]
                audio_file_path = os.path.join(OUTPUT_DIR, f"{i}{ext}")

                try:
                    import shutil
                    shutil.copy(src_path, audio_file_path)
                except:
                    audio_file_path = src_path  # fallback

        # ====== CASE 2: audio là string path ======
        elif isinstance(audio, str):
            ext = os.path.splitext(audio)[-1]
            audio_file_path = os.path.join(OUTPUT_DIR, f"{i}{ext}")

            try:
                import shutil
                shutil.copy(audio, audio_file_path)
            except:
                audio_file_path = audio

        # write transcript
        f.write(f"[{i}] AUDIO: {audio_file_path}\n")
        f.write(f"TEXT: {text}\n")
        f.write("-" * 50 + "\n")


print("Done! Saved to:", OUTPUT_DIR)