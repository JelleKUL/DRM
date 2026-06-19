import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import drm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../Hunyuan3D-Omni/')))
import inference

def check_torch():
    import torch
    print("torch is installed:")
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("CUDA device count:", torch.cuda.device_count())
    else:
        print("No CUDA GPU detected")
    print("torch version: " + torch.__version__)
    print("torch-cuda version: " + torch.version.cuda)