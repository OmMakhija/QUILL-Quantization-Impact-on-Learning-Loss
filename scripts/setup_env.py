"""
Hardware check, environment setup, and OpenUnlearning verification script.
"""
import sys
import os
import torch

def check_environment():
    print("=== Environment Verification ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        bf16_sup = torch.cuda.is_bf16_supported()
        print(f"GPU: {gpu_name} ({gpu_mem:.2f} GB)")
        print(f"BF16 Supported: {bf16_sup}")
    else:
        print("[!] Warning: CUDA is not available. GPU is recommended for training and evaluation.")

def create_project_structure(base_dir="./quill_artifacts"):
    subdirs = ["checkpoints", "results", "logs"]
    for sub in subdirs:
        path = os.path.join(base_dir, sub)
        os.makedirs(path, exist_ok=True)
        print(f"Ready: {path}")

if __name__ == "__main__":
    check_environment()
    create_project_structure()