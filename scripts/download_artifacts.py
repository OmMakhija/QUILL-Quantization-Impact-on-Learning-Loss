"""
Script to download TOFU dataset splits and pre-trained fine-tuned model weights.
"""
import os
import argparse
from datasets import load_dataset
from huggingface_hub import snapshot_download

def verify_tofu_splits():
    splits = [
        "full",
        "forget01", "retain99",
        "forget05", "retain95",
        "forget10", "retain90"
    ]
    print("=== Checking TOFU Dataset Splits ===")
    for split_name in splits:
        try:
            ds = load_dataset("locuslab/TOFU", split_name)
            print(f"Split: {split_name:<10} | Samples: {len(ds['train'])}")
        except Exception as e:
            print(f"Error loading {split_name}: {e}")

def download_model(repo_id="locuslab/tofu_ft_phi-1.5", target_dir="./quill_artifacts/checkpoints/phi15_tofu_ft"):
    os.makedirs(target_dir, exist_ok=True)
    print(f"=== Downloading {repo_id} to {target_dir} ===")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin"]
        )
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Failed to download model: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TOFU dataset and base checkpoints.")
    parser.add_argument("--skip-data", action="store_true", help="Skip dataset split verification.")
    parser.add_argument("--skip-model", action="store_true", help="Skip downloading fine-tuned checkpoint.")
    parser.add_argument("--model-dir", default="./quill_artifacts/checkpoints/phi15_tofu_ft", help="Path to store checkpoint.")
    args = parser.parse_args()

    if not args.skip_data:
        verify_tofu_splits()
    if not args.skip_model:
        download_model(target_dir=args.model_dir)