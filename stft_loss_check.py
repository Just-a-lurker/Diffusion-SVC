import librosa
import numpy as np

def calculate_stft_loss(file_generated, file_real, n_fft=1024, hop_length=256):
    # Nạp audio
    y_gen, sr = librosa.load(file_generated, sr=None)
    y_real, _ = librosa.load(file_real, sr=sr)

    # Đảm bảo 2 file có cùng độ dài (cắt phần thừa nếu chênh lệch vài sample)
    min_len = min(len(y_gen), len(y_real))
    y_gen = y_gen[:min_len]
    y_real = y_real[:min_len]

    # Tính toán STFT
    stft_gen = np.abs(librosa.stft(y_gen, n_fft=n_fft, hop_length=hop_length))
    stft_real = np.abs(librosa.stft(y_real, n_fft=n_fft, hop_length=hop_length))

    # 1. Spectral Convergence Loss
    spectral_convergence = np.linalg.norm(stft_real - stft_gen, ord='fro') / np.linalg.norm(stft_real, ord='fro')

    # 2. Log STFT Magnitude Loss
    # Thêm epsilon (1e-7) để tránh lỗi log(0)
    log_stft_loss = np.mean(np.abs(np.log(np.clip(stft_real, a_min=1e-7, a_max=None)) - 
                                   np.log(np.clip(stft_gen, a_min=1e-7, a_max=None))))

    return spectral_convergence, log_stft_loss

# --- Thay đường dẫn của bạn vào đây ---
file_ground_truth = "data/temp/test5m.mp3"
file_vocoder_cu = "data/out_oldhfg.wav"
file_vocoder_moi = "data/out.wav"

sc_cu, log_cu = calculate_stft_loss(file_vocoder_cu, file_ground_truth)
sc_moi, log_moi = calculate_stft_loss(file_vocoder_moi, file_ground_truth)

print(f"--- VOCODER CŨ ---")
print(f"Spectral Convergence: {sc_cu:.5f}")
print(f"Log STFT Loss:        {log_cu:.5f}\n")

print(f"--- VOCODER MỚI (FINETUNED) ---")
print(f"Spectral Convergence: {sc_moi:.5f}")
print(f"Log STFT Loss:        {log_moi:.5f}")