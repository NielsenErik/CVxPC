import os
import json
import pandas as pd
from glob import glob
from collections import defaultdict

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------
log_root = "logs/old/pcnet_ge"   # ← CHANGE THIS to your logs root folder
output_csv = "experiment_summary.csv"

# ----------------------------------------------------------
# Helper to load a single experiment.json
# ----------------------------------------------------------
def load_experiment_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

# ----------------------------------------------------------
# Collect all experiment files grouped by dataset
# Folder structure:
#   algorithm/
#       dataset/
#           seed/
#               experiment.json
# ----------------------------------------------------------
def collect_experiment_files(root):
    experiment_files = defaultdict(list)

    # pattern: log_root/*/*/*/experiment.json
    all_jsons = glob(os.path.join(root, "**", "experiment.json"), recursive=True)
    print(f"Found {len(all_jsons)} experiment.json files.")
    for json_path in all_jsons:
        parts = json_path.split(os.sep)

        # Extract algorithm, dataset, seed
        # [..., log_root, algorithm, dataset, seed, experiment.json]
        algorithm = parts[-4]
        dataset = parts[-3]
        seed = parts[-2]

        key = dataset  # group by dataset only
        experiment_files[key].append(json_path)

    return experiment_files

# ----------------------------------------------------------
# Extract all metrics from an experiment.json into flat dict
# ----------------------------------------------------------
def extract_metrics(exp):
    metrics = {}

    # Pull training metrics
    train = exp.get("training", {})
    for k, v in train.items():
        metrics[f"train_{k}"] = v

    # Pull test results
    results = exp.get("results", {})
    for k, v in results.items():
        metrics[f"{k}"] = v

    return metrics


# ----------------------------------------------------------
# Compute summary statistics for a list of metric dicts
# ----------------------------------------------------------
def summarize_metrics(metric_dicts):
    df = pd.DataFrame(metric_dicts)

    summary = {"dataset": None}
    for col in df.columns:
        summary[f"{col}_mean"] = df[col].mean()
        summary[f"{col}_std"] = df[col].std()
        summary[f"{col}_best"] = df[col].max()  # best = highest score
    return summary


# ----------------------------------------------------------
# Main script
# ----------------------------------------------------------
def main(log_root=log_root, output_csv=output_csv):
    experiment_files = collect_experiment_files(log_root)

    summaries = []

    for dataset, files in experiment_files.items():
        print(f"\nProcessing dataset: {dataset} ({len(files)} seeds)")

        metrics_list = []

        for json_path in files:
            exp = load_experiment_json(json_path)
            if exp is None:
                continue

            metrics = extract_metrics(exp)
            metrics_list.append(metrics)

        if not metrics_list:
            print(f"No valid experiments for dataset {dataset}")
            continue

        summary = summarize_metrics(metrics_list)
        summary["dataset"] = dataset
        summaries.append(summary)

    # final combined CSV
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(output_csv, index=False)

    print(f"\nSummary CSV saved to: {output_csv}")


if __name__ == "__main__":
    log_root = "logs/old/pcnet_ge"   # ← CHANGE THIS to your logs root folder
    output_csv = "logs/old/pcnet_ge/experiment_summary.csv"
    main(log_root, output_csv)
