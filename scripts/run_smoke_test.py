"""
Runs a minimal 2-step unlearning smoke test using OpenUnlearning.
"""
import os
import subprocess
import argparse

def run_smoke_test(open_unlearning_dir="./open-unlearning", out_dir="./quill_artifacts/logs/smoke_test"):
    os.makedirs(out_dir, exist_ok=True)
    
    cmd = [
        "python", "src/train.py",
        "--config-name=unlearn.yaml",
        "experiment=unlearn/tofu/default",
        "model=phi-1_5",
        "++model.model_args.attn_implementation=sdpa",
        "forget_split=forget01",
        "retain_split=retain99",
        "trainer=GradAscent",
        "task_name=SMOKE_TEST_RUN",
        f"hydra.run.dir={out_dir}",
        "+trainer.max_steps=2",
        "+trainer.fp16=True",
        "+trainer.bf16=False",
        "+trainer.gradient_accumulation_steps=1",
        "+trainer.per_device_train_batch_size=1"
    ]
    
    print(f"Running smoke test in {open_unlearning_dir}...")
    subprocess.run(cmd, cwd=open_unlearning_dir, check=True)
    print("Smoke test finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default="./open-unlearning", help="Path to open-unlearning repo")
    parser.add_argument("--out-dir", default="./quill_artifacts/logs/smoke_test", help="Output directory")
    args = parser.parse_args()
    
    run_smoke_test(open_unlearning_dir=args.repo_dir, out_dir=args.out_dir)