import json
from pathlib import Path
import matplotlib.pyplot as plt


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
    for _, cfg in models.items():
        cfg["loss"] = loss_load_json(cfg["loss_path"])
        cfg["epochs"] = list(range(1, len(cfg["loss"])+1))
        # Loss history loading

    plt.figure(figsize=(10, 5))
    for name, cfg in models.items():
        plt.plot(cfg["epochs"], cfg["loss"], label=name)
        plt.xlabel("Epochs")
        plt.ylabel("Training Loss")
        plt.title("Training Loss Per Epoch For Models")
        plt.legend()
        plt.grid(True, linestyle=":", linewidth=0.5)
        # Training loss plotting

    plt.figure(figsize=(10, 5))
    model_names = list(models.keys())
    accuracies = [models[m]["test_accuracy"] for m in model_names]
    x = range(len(model_names))
    plt.bar(x, accuracies)
    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()
    # Accuracy bar plot


if __name__ == "__main__":
    main()
