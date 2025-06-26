import torch
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
import cv2

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

print("NumPy version:", np.__version__)
print("Pandas version:", pd.__version__)

# 测试GPU内存
if torch.cuda.is_available():
    x = torch.randn(100, 100, 100).cuda()
    print("GPU test successful!")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**2:.1f} MB")