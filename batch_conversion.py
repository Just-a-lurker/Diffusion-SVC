import os
import subprocess
import sys

# --- CẤU HÌNH CÁC THAM SỐ CỐ ĐỊNH TẠI ĐÂY ---
# Nếu không muốn dùng tham số nào, hãy đặt giá trị là "none" hoặc None
MODEL_CKPT = "exp/16_1_Pass3_rmvpe_10h_each_hifigan2024_nofinetune/diff/model_90000.pt"
SPEAKER_ID = "2"
METHOD = "dpm-solver"
KSTEP = "100"
NMODEL = "exp/16_1_Pass3_rmvpe_10h_each_hifigan2024_nofinetune/naive/model_50000.pt"  # Tham số này có giá trị là "none" nên sẽ bị loại bỏ khỏi câu lệnh
MANUAL_SEMITONE = "none"
PITCH_EXTRACTOR = "rmvpe"

INPUT_FOLDER = "./test"
OUTPUT_FOLDER = "./output_wavs"
# --------------------------------------------

def run_batch():
    # Tạo thư mục đầu ra nếu chưa tồn tại
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Đã tạo thư mục đầu ra: {OUTPUT_FOLDER}")

    # Lấy danh sách file trong thư mục đầu vào
    try:
        files = os.listdir(INPUT_FOLDER)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy thư mục đầu vào '{INPUT_FOLDER}'")
        return

    # Lọc file .wav
    wav_files = [f for f in files if f.lower().endswith('.wav')]

    if not wav_files:
        print("Không tìm thấy file .wav nào trong thư mục đầu vào.")
        return

    print(f"Tìm thấy {len(wav_files)} file wav cần xử lý.\n")

    for index, file_name in enumerate(wav_files, 1):
        input_path = os.path.join(INPUT_FOLDER, file_name)
        output_path = os.path.join(OUTPUT_FOLDER, f"processed_{file_name}")

        print(f"[{index}/{len(wav_files)}] Đang xử lý: {file_name}")

        # 1. Khởi tạo câu lệnh với các tham số bắt buộc ban đầu
        cmd = [sys.executable, "main.py", "-i", input_path, "-o", output_path]

        # 2. Gom nhóm các tham số tùy chọn vào một dictionary
        optional_params = {
            "-model": MODEL_CKPT,
            "-id": SPEAKER_ID,
            "-method": METHOD,
            "-kstep": KSTEP,
            "-nmodel": NMODEL,
            "-mn": MANUAL_SEMITONE,
            "-pe": PITCH_EXTRACTOR
        }

        # 3. Duyệt qua từng tham số, chỉ thêm vào cmd nếu giá trị khác "none" hoặc None
        for flag, value in optional_params.items():
            if value is not None and str(value).strip().lower() != "none":
                cmd.extend([flag, str(value)])

        # Bỏ ghi chú dòng dưới đây nếu bạn muốn kiểm tra xem câu lệnh sinh ra trông như thế nào:
        # print("Lệnh thực thi:", " ".join(cmd))

        # Thực thi câu lệnh
        print(cmd)
        try:
            subprocess.run(cmd, check=True)
            print(f"-> Thành công! Đã lưu vào: {output_path}\n")
        except subprocess.CalledProcessError as e:
            print(f"-> Lỗi khi xử lý file {file_name}: {e}\n")

    print("Đã hoàn thành xử lý toàn bộ thư mục!")

if __name__ == "__main__":
    run_batch()