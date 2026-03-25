import os
import random
import shutil

def copy_random_files(folder1, folder2, x):
    # Tạo folder2 nếu chưa tồn tại
    os.makedirs(folder2, exist_ok=True)

    # Lấy danh sách file
    all_files = [
        f for f in os.listdir(folder1)
        if os.path.isfile(os.path.join(folder1, f))
    ]

    if x > len(all_files):
        print("Số lượng yêu cầu lớn hơn số file hiện có!")
        return

    # Chọn ngẫu nhiên x file
    selected_files = random.sample(all_files, x)

    # Copy file
    for file in selected_files:
        src_path = os.path.join(folder1, file)
        dst_path = os.path.join(folder2, file)
        shutil.copy2(src_path, dst_path)  # copy giữ metadata

    print(f"Đã copy {x} file sang {folder2}")


# Ví dụ sử dụng
folder1 = "D:\Tools\Projects\Diffusion-SVC\data/train/audio/2"
folder2 = "D:\Tools\Projects\Diffusion-SVC\data/train/audio/testfile/2"
x = 10

copy_random_files(folder1, folder2, x)