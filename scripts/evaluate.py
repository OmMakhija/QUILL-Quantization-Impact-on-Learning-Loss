import os
import subprocess
import argparse

def run_evaluation(
    model_path="./quill_artifacts/checkpoints/phi15_tofu_ft",
    forget_split="forget05",
    retain_split="retain95",
    repo_dir="./open-unlearning",
    output_dir="./quill_artifacts/results"
):
    run_dir = os.path.join(output_dir, f"eval_{forget_split}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        "python", "src/eval.py",
        "--config-name=eval.yaml",
        "experiment=eval/tofu/default",
        "model=phi-1_5",
        f"model.model_args.pretrained_model_name_or_path={os.path.abspath(model_path)}",
        "++model.model_args.attn_implementation=sdpa",
        "++model.model_args.device_map=cpu",
        "model.tokenizer_args.pretrained_model_name_or_path=microsoft/phi-1_5",
        f"forget_split={forget_split}",
        f"+retain_split={retain_split}",
        "+trainer.fp16=False",
        "+trainer.bf16=False",
        f"+task_name=eval_{forget_split}",
        f"hydra.run.dir={os.path.abspath(run_dir)}"
    ]

    print(f"=== Evaluating {forget_split} with base {model_path} ===")
    print("Running command:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_dir, check=True)
    print(f"Evaluation complete for {forget_split}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TOFU evaluation.")
    parser.add_argument("--model-path", default="./quill_artifacts/checkpoints/phi15_tofu_ft", help="Path to model checkpoint")
    parser.add_argument("--forget-split", default="forget05", choices=["forget01", "forget05", "forget10"])
    parser.add_argument("--retain-split", default="retain95", choices=["retain99", "retain95", "retain90"])
    parser.add_argument("--repo-dir", default="./open-unlearning", help="Path to open-unlearning repository")
    parser.add_argument("--output-dir", default="./quill_artifacts/results", help="Directory where results will be written")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        forget_split=args.forget_split,
        retain_split=args.retain_split,
        repo_dir=args.repo_dir,
        output_dir=args.output_dir
    )