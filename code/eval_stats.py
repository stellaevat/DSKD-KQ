import sys
import re
import statistics
from collections import defaultdict
from pathlib import Path

DATASET_ORDER = ["dolly", "self-inst", "vicuna", "sinst/11_", "uinst/11_"]

def extract_scores(filepath):
    rougeL_scores = defaultdict(list)
    em_scores = defaultdict(list)

    # Regex to capture dataset name, exact_match, and rougeL
    pattern = re.compile(
        r"name:\s*(.*?)\s*\|\s*{\s*'exact_match':\s*([\d.]+),\s*'rougeL':\s*([\d.]+)\s*}"
    )

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                dataset = match.group(1).strip()
                em = float(match.group(2))
                rougeL = float(match.group(3))
                em_scores[dataset].append(em)
                rougeL_scores[dataset].append(rougeL)

    return rougeL_scores, em_scores

def compute_stats(scores_dict, output_path):
    means = []
    ood_means = []
    lines = []

    header = f"{'Dataset':12} | {'Mean':>8} | {'Std Dev':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    for dataset in DATASET_ORDER:
        scores = scores_dict.get(dataset, [])
        if not scores:
            print(f"Warning: Dataset '{dataset}' not found in log.")
            continue
        if len(scores) != 5:
            print(f"Warning: Dataset '{dataset}' has {len(scores)} scores (expected 5).")
        mean = statistics.mean(scores)
        stdev = statistics.stdev(scores)
        means.append(mean)
        if dataset != "dolly":
            ood_means.append(mean)
        lines.append(f"{dataset:12} | {mean:8.2f} | {stdev:8.2f}")

    avg_mean = statistics.mean(means)
    avg_ood_mean = statistics.mean(ood_means) if ood_means else 0

    lines.append("-" * len(header))
    lines.append(f"{'OOD Average':12} | {avg_ood_mean:8.2f}")
    lines.append(f"{'Average Mean':12} | {avg_mean:8.2f}")

    return lines

def compute_and_write_stats(rougeL_scores, em_scores, output_path):
    all_lines = []
    
    # Rouge-L table
    all_lines.extend(compute_stats(rougeL_scores, "Rouge-L"))
    all_lines.append("")  # blank line between tables

    # Exact Match table
    all_lines.extend(compute_stats(em_scores, "Exact Match"))

    with open(output_path, 'w') as f:
        f.write("\n".join(all_lines))

    print(f"Stats written to: {output_path}")

def main():
    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"Error: File not found: {input_path}")
        return

    output_path = input_path.parent / "stats.txt"
    rougeL_scores, em_scores = extract_scores(input_path)
    compute_and_write_stats(rougeL_scores, em_scores, output_path)

if __name__ == "__main__":
    main()
