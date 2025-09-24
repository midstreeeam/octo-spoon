import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)                # should be a non-None string on AMD
print("cuda available:", torch.cuda.is_available())  # True on ROCm too
print("device name:", torch.cuda.get_device_name(0))