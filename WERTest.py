import re
from pathlib import Path
import librosa
import whisper
from jiwer import wer
from tqdm import tqdm


# CONFIG

WHISPER_MODEL = "large-v3"

# File transcript:
TRANSCRIPT_FILE = "E:\Projects\Diffusion-SVC/test/transcripts.txt"

# Thư mục chứa audio VC output
AUDIO_DIR = "E:\Projects\Diffusion-SVC/Male"


# TEXT NORMALIZATION


def normalize(text: str) -> str:
    text = text.lower()

    # bỏ dấu câu
    text = re.sub(r"[^\w\s]", " ", text)

    # chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text)

    return text.strip()



# LOAD TRANSCRIPTS


def load_transcripts(path):
    transcripts = {}

    with open(path, "r", encoding="utf-8") as f:
        current_id = None

        for line in f:
            line = line.strip()

            # Bỏ qua dòng trống hoặc dòng gạch ngang phân cách
            if not line or line.startswith("---"):
                continue

            # Nếu dòng chứa "AUDIO:", trích xuất tên file
            if "AUDIO:" in line:
                # Tách chuỗi để lấy phần sau chữ AUDIO: (ví dụ: "test\0.wav")
                raw_audio_path = line.split("AUDIO:")[1].strip()

                # Trích xuất riêng tên file không chứa đuôi (ví dụ: "0") làm ID
                # Path(raw_audio_path).stem sẽ loại bỏ đường dẫn thư mục và đuôi .wav
                current_id = Path(raw_audio_path).stem

            # Nếu dòng bắt đầu bằng "TEXT:" và đã có ID từ bước trước
            elif line.startswith("TEXT:"):
                if current_id is not None:
                    # Lấy toàn bộ nội dung sau chữ "TEXT:"
                    text_content = line.split("TEXT:", 1)[1].strip()
                    transcripts[current_id] = text_content

                    # Reset lại current_id để xử lý block tiếp theo
                    current_id = None

    return transcripts



# MAIN


def main():

    print("Loading Whisper...")
    model = whisper.load_model(WHISPER_MODEL, device="cpu")

    transcripts = load_transcripts(TRANSCRIPT_FILE)

    refs = []
    preds = []

    sample_wers = []

    print(f"Found {len(transcripts)} utterances")

    for utt_id, gt_text in tqdm(transcripts.items()):

        wav_path = Path(AUDIO_DIR) / f"{utt_id}.wav"

        if not wav_path.exists():
            print(f"[WARNING] Missing file: {wav_path}")
            continue

        y, sr = librosa.load(wav_path, sr=16000)

        # cắt silence đầu/cuối
        y, _ = librosa.effects.trim(y, top_db=25)
        result = model.transcribe(
            y,
            language="vi",
            task="transcribe",
            fp16=True,
            condition_on_previous_text=False
        )

        pred_text = result["text"]

        ref = normalize(gt_text)
        pred = normalize(pred_text)

        refs.append(ref)
        preds.append(pred)

        sample_wer = wer(ref, pred)

        sample_wers.append(
            {
                "utt_id": utt_id,
                "wer": sample_wer,
                "ref": ref,
                "pred": pred,
            }
        )

    # CORPUS WER

    corpus_wer = wer(refs, preds)

    # Sắp xếp các file từ WER cao nhất đến thấp nhất
    sample_wers.sort(
        key=lambda x: x["wer"],
        reverse=True
    )

    # XUẤT KẾT QUẢ RA FILE TXT

    output_txt_path = Path(AUDIO_DIR) / "wer_results.txt"

    with open(output_txt_path, "w", encoding="utf-8") as f_out:
        # Ghi kết quả chi tiết của từng file
        for item in sample_wers:
            f_out.write(f"ID  : {item['utt_id']}\n")
            f_out.write(f"WER : {item['wer'] * 100:.2f}%\n")
            f_out.write(f"REF : {item['ref']}\n")
            f_out.write(f"PRED: {item['pred']}\n")
            f_out.write("-" * 50 + "\n")  # Dòng kẻ phân cách giữa các file

        # Ghi WER trung bình vào dòng cuối cùng
        f_out.write(f"\nWER trung bình: {corpus_wer * 100:.2f}%\n")

    # Vẫn giữ print ra màn hình console để bạn theo dõi nhanh
    print("\n==============================")
    print(f"Corpus WER: {corpus_wer * 100:.2f}%")
    print(f"Đã lưu kết quả chi tiết vào: {output_txt_path}")
    print("==============================")



if __name__ == "__main__":
    main()