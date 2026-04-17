import librosa
import numpy as np
from pystoi import stoi
from pesq import pesq


def calculate_mcd(y_gen, y_real, sr, n_mfcc=13):
    mfcc_gen = librosa.feature.mfcc(y=y_gen, sr=sr, n_mfcc=n_mfcc)[1:, :]
    mfcc_real = librosa.feature.mfcc(y=y_real, sr=sr, n_mfcc=n_mfcc)[1:, :]

    diff = mfcc_real - mfcc_gen
    mcd_frames = (10.0 / np.log(10.0)) * np.sqrt(2.0 * np.sum(diff ** 2, axis=0))
    return np.mean(mcd_frames)


def calculate_f0_rmse(y_gen, y_real, sr):
    f0_gen, _, _ = librosa.pyin(y_gen, fmin=65, fmax=2000, sr=sr)
    f0_real, _, _ = librosa.pyin(y_real, fmin=65, fmax=2000, sr=sr)

    mask = ~np.isnan(f0_gen) & ~np.isnan(f0_real)
    if not np.any(mask):
        return 0.0

    f0_gen_voiced = f0_gen[mask]
    f0_real_voiced = f0_real[mask]

    cents_gen = 1200 * np.log2(f0_gen_voiced / 10.0)
    cents_real = 1200 * np.log2(f0_real_voiced / 10.0)

    rmse_cents = np.sqrt(np.mean((cents_real - cents_gen) ** 2))
    return rmse_cents


def evaluate_audio_quality(file_generated, file_real, n_fft=1024, hop_length=256):
    # 1. Nạp audio và đồng bộ Sample Rate
    y_gen, sr_gen = librosa.load(file_generated, sr=None)
    y_real, sr_real = librosa.load(file_real, sr=None)

    if sr_gen != sr_real:
        y_real = librosa.resample(y_real, orig_sr=sr_real, target_sr=sr_gen)
    sr = sr_gen

    # 2. Cắt phần thừa để 2 mảng có cùng kích thước
    min_len = min(len(y_gen), len(y_real))
    y_gen = y_gen[:min_len]
    y_real = y_real[:min_len]

    # --- TÍNH TOÁN CÁC METRICS ---

    # 1. STFT LOSS
    stft_gen = np.abs(librosa.stft(y_gen, n_fft=n_fft, hop_length=hop_length))
    stft_real = np.abs(librosa.stft(y_real, n_fft=n_fft, hop_length=hop_length))

    spectral_convergence = np.linalg.norm(stft_real - stft_gen, ord='fro') / np.linalg.norm(stft_real, ord='fro')
    log_stft_loss = np.mean(np.abs(np.log(np.clip(stft_real, a_min=1e-7, a_max=None)) -
                                   np.log(np.clip(stft_gen, a_min=1e-7, a_max=None))))

    # 2. MCD và F0 RMSE
    mcd = calculate_mcd(y_gen, y_real, sr)
    f0_rmse = calculate_f0_rmse(y_gen, y_real, sr)

    # 3. STOI (Short-Time Objective Intelligibility)
    # Hàm stoi nhận: tham chiếu, tín hiệu biến dạng, sample rate
    stoi_score = stoi(y_real, y_gen, sr, extended=False)

    # 4. PESQ (Perceptual Evaluation of Speech Quality)
    # Bắt buộc resample về 16000Hz (Wideband) để tính PESQ
    if sr != 16000:
        y_real_16k = librosa.resample(y_real, orig_sr=sr, target_sr=16000)
        y_gen_16k = librosa.resample(y_gen, orig_sr=sr, target_sr=16000)
    else:
        y_real_16k = y_real
        y_gen_16k = y_gen

    try:
        # 'wb' là Wideband. Truyền đúng thứ tự: (sr, tham chiếu, tín hiệu biến dạng, mode)
        pesq_score = pesq(16000, y_real_16k, y_gen_16k, 'wb')
    except Exception as e:
        # PESQ sẽ quăng lỗi NoUtterancesError nếu file âm thanh toàn là khoảng lặng (silence)
        pesq_score = float('nan')

    return {
        "Spectral Convergence": spectral_convergence,
        "Log STFT Loss": log_stft_loss,
        "MCD": mcd,
        "F0 RMSE (Cents)": f0_rmse,
        "STOI": stoi_score,
        "PESQ (Wideband)": pesq_score
    }


# --- Chạy thử ---
file_ground_truth = "D:/Tools\Projects\Diffusion-SVC\data/test/testLong/test5mc.mp3"
file_vocoder_cu = "D:/Tools\Projects\Diffusion-SVC\data/test/testLong/old_female.wav"
file_vocoder_moi = "D:/Tools\Projects\Diffusion-SVC\data/test/testLong/new_female.wav"
# file_vocoder_moi2 = "D:/Tools\Projects\Diffusion-SVC\data/test/246k/5637_6k.wav"

metrics_cu = evaluate_audio_quality(file_vocoder_cu, file_ground_truth)
metrics_moi = evaluate_audio_quality(file_vocoder_moi, file_ground_truth)
# metrics_moi2 = evaluate_audio_quality(file_vocoder_moi2, file_ground_truth)

print("--- 2  ---")
for k, v in metrics_cu.items():
    print(f"{k:<25}: {v:.5f}" if not np.isnan(v) else f"{k:<25}: Lỗi xử lý (silence)")

print("\n--- 4 ---")
for k, v in metrics_moi.items():
    print(f"{k:<25}: {v:.5f}" if not np.isnan(v) else f"{k:<25}: Lỗi xử lý (silence)")

# print("\n--- 6 ---")
# for k, v in metrics_moi2.items():
#     print(f"{k:<25}: {v:.5f}" if not np.isnan(v) else f"{k:<25}: Lỗi xử lý (silence)")