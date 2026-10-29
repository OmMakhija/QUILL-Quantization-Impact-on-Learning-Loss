"""
Reads and displays summary evaluation results from OpenUnlearning output JSON files.
"""
import os
import json
import glob
import argparse

def display_summary(json_path, split_name="TOFU Split"):
    print(f"\n{'='*25} {split_name} {'='*25}")
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"Model Utility               : {data.get('model_utility', 'N/A')}")
    print(f"Forget ROUGE                : {data.get('forget_Q_A_ROUGE', data.get('forget_rouge', 'N/A'))}")
    print(f"Extraction Strength (Forget): {data.get('extraction_strength', 'N/A')}")
    print(f"Extraction Strength (Retain): {data.get('retain_extraction_strength', 'N/A')}")
    print(f"PrivLeak                    : {data.get('privleak', 'N/A')}")

    fq = data.get('forget_quality', None)
    if fq is not None:
        print(f"Forget Quality              : {fq}")
    else:
        print("Forget Quality              : Not computed (requires reference retain model logs)")

def scan_results(search_dirs):
    print("Searching for evaluation summary files...")
    found = False
    for base in search_dirs:
        if os.path.exists(base):
            summaries = glob.glob(os.path.join(base, "**", "*SUMMARY*.json"), recursive=True)
            for s in summaries:
                found = True
                split = os.path.basename(os.path.dirname(s))
                display_summary(s, split_name=split)
    if not found:
        print("No evaluation summary files found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None, help="Specific TOFU_SUMMARY.json to parse")
    parser.add_argument("--scan-dirs", nargs="+", default=["./quill_artifacts/results", "./open-unlearning/saves/eval"], help="Directories to scan for summaries")
    args = parser.parse_args()

    if args.json:
        display_summary(args.json)
    else:
        scan_results(args.scan_dirs)