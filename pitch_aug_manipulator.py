import numpy as np
from pathlib import Path
from typing import Dict

NPY_PATH = r"D:\Tools\Projects\Diffusion-SVC\data\val\pitch_aug_dict.npy"
DATA_DIR = Path(r"D:\Tools\Projects\Diffusion-SVC\data\val\audio")

data = np.load(NPY_PATH, allow_pickle=True).item()

print("Total entries before:", len(data))

removed = 0
to_delete = []

for i, (k, v) in enumerate(data.items()):
    print(f"{i}: key =", k)
    print("value =", v)
    print("-" * 40)
    if i >= 11:
        break


# for key in data.keys():
#     wav_path = DATA_DIR / key
#     # print(wav_path)
#     if not wav_path.exists():
#         to_delete.append(key)
#
# for key in to_delete:
#     del data[key]
#     print("REMOVED:", key)
#     removed += 1
#
# print("Total removed entries:", removed)
# print("Total entries after:", len(data))
#
# np.save(NPY_PATH, data)

# new_data = {}
# renamed = 0
#
# for key, value in data.items():
#     if key.startswith("1\\"):
#         new_key = key.replace("1\\", "2\\", 1)
#         print(f"{key} -> {new_key}")
#         new_data[new_key] = value
#         renamed += 1
#     else:
#         new_data[key] = value
#
# print("Total renamed:", renamed)
# np.save(NPY_PATH, new_data)

# def add_entries_from_old_file(old_npy_path, new_npy_path, overwrite=False):
#     """
#     Thêm các entry từ file cũ vào file mới
#
#     overwrite=False: không ghi đè nếu key đã tồn tại
#     overwrite=True : ghi đè value nếu key trùng
#     """
#
#     old_data = np.load(old_npy_path, allow_pickle=True).item()
#     new_data = np.load(new_npy_path, allow_pickle=True).item()
#
#     added = 0
#     skipped = 0
#     overwritten = 0
#
#     for key, value in old_data.items():
#         if key in new_data:
#             if overwrite:
#                 new_data[key] = value
#                 overwritten += 1
#             else:
#                 skipped += 1
#         else:
#             new_data[key] = value
#             added += 1
#
#     print("Added:", added)
#     print("Skipped (existing):", skipped)
#     print("Overwritten:", overwritten)
#     print("Total entries after:", len(new_data))
#
#     np.save(new_npy_path, new_data)
#
#
# add_entries_from_old_file(
#     old_npy_path="pitch_aug_dict_1.npy",
#     new_npy_path="pitch_aug_dict_2.npy",
#     overwrite=False
# )
