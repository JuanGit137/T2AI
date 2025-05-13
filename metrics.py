import json
import matplotlib.pyplot as plt
import os

# Load JSON
with open("results/genresults/evaluation_results.json", "r") as f:
    data = json.load(f)

# Group models by dataset prefix
datasets = {}
for key, val in data.items():
    dataset = val["dataset"]
    if dataset not in datasets:
        datasets[dataset] = []
    datasets[dataset].append(val)

# Loop over each dataset
for dataset, models_data in datasets.items():
    # --- mAP Bar Plot ---
    model_names = [m['model'] for m in models_data]
    mAPs = [m['mAP'] for m in models_data]

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, mAPs, color='skyblue', width=0.4)
    plt.title(f'mAP Comparison - {dataset}')
    plt.ylabel('Mean Average Precision')
    plt.xlabel('Model')
    plt.ylim(0, 1)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Precision-Recall Curve ---
    plt.figure(figsize=(8, 5))
    for m in models_data:
        pr_data = m['precision_recall']['class_data']['average']
        recalls = pr_data['recalls']
        precisions = pr_data['precisions']
        plt.plot(recalls, precisions, label=m['model'])

    plt.title(f'Average Precision-Recall Curve - {dataset}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
