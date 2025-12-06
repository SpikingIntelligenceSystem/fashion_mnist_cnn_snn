import json
from pathlib import Path
import matplotlib.pyplot as plt
import os


def loss_load_json(path):
    path = Path(path)
    with open(path, "r") as f:
        loss_list = json.load(f)
    return loss_list


def main():
    models = {
        "CNN": {
            "loss_path": "results/cnn_loss.json",
            "test_accuracy": 92.12,

        },
        "CSNN Single Step": {
            "loss_path": "results/csnn_single_loss.json",
            "test_accuracy": 90.65,
        },
        "CSNN Temporal": {
            "loss_path": "results/csnn_temporal_loss.json",
            "test_accuracy": 89.41,
        },
        # Copy recorded accuracies from training in respective areas
    }
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Establish results directory

    for _, cfg in models.items():
        cfg["loss"] = loss_load_json(cfg["loss_path"])
        cfg["epochs"] = list(range(1, len(cfg["loss"])+1))
        # Loss history loading

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    for name, cfg in models.items():
        ax1.plot(cfg["epochs"], cfg["loss"], label=name)
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss Per Epoch For Models")
        ax1.legend()
        ax1.grid(True, linestyle=":", linewidth=0.5)

        loss_path = results_dir / "training_loss_comparison.png"
        fig1.savefig(loss_path, dpi=300, bbox_inches="tight")
        print(f"Saved loss line to: {loss_path}")
        # Training loss plotting

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    model_names = list(models.keys())
    accuracies = [models[m]["test_accuracy"] for m in model_names]
    x = range(len(model_names))
    ax2.bar(x, accuracies)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(model_names, rotation=15, ha="right")
    ax2.set_ylabel("Test Accuracy Percentage")
    ax2.set_title("Accuracy Comparison Between Models")
    ax2.set_ylim(0, 100)
    ax2.grid(axis="y", linestyle=":", linewidth=0.5)
    # Accuracy bar plot

    acc_path = results_dir / "test_accuracy_comparison.png"
    fig2.savefig(acc_path, dpi=300, bbox_inches="tight")
    print(f"Saved accuracy graph to: {acc_path}")

    plt.tight_layout()


if __name__ == "__main__":
    main()
