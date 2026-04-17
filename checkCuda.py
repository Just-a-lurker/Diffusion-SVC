import torch
print(torch.cuda.is_available())  # Phải trả về True
print(torch.version.cuda)         # Xem phiên bản CUDA PyTorch đang dùng
print(torch.cuda.device_count())  # Số GPU nhận diện được
print(torch.cuda.get_device_name(0))  # Tên GPU
print(torch.__version__)
# import FreeSimpleGUI as sg
# print(dir(sg))
