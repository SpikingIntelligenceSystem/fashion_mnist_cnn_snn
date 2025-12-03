# fashion_mnist_cnn_snn
## About
Supervised image classification project on **Fashion-MNIST**, comparing a
standard convolutional neural network (CNN) baseline against spiking
convolutional neural networks (SNNs) implemented with [snnTorch](https://github.com/jeshraghian/snntorch).

---

## Project Overview

This repository extends my earlier MNIST SNN project to the **Fashion-MNIST**
dataset. The goal is to:

- Build a strong **CNN baseline** for Fashion-MNIST.
- Implement **single-step** and **temporal** spiking CNN models.
- Compare performance, training dynamics, and implementation details between
  standard CNNs and SNNs.

---

## Primary Personal Outcomes

Working through this project, I aim to:

- Continue practice in structuring a PyTorch project with clear separation between
  **models**, **training scripts**, and **result scripts**.
- Understand how to implement and train:
  - A standard CNN classifier for Fashion-MNIST.
  - Spiking CNNs using snnTorch (LIF layers, temporal dynamics).
- Compare **accuracy**, **training stability**, and **inference behavior**
  between CNN and SNN approaches.
- Build a reusable template for future CNN/SNN classification projects.

---

## Repository Structure

```text
FASHION_MNIST_CNN_SNN/
├─ models/
│  ├─ cnn_baseline.py            # FashionCNN: standard CNN classifier
│  ├─ csnn_single_step.py        # CSNNSingleStep: single-step spiking CNN
│  └─ csnn_temporal.py           # CSNNTemporal: temporal spiking CNN
│
├─ train_scripts/
│  ├─ train_cnn.py               # Train/evaluate CNN baseline
│  ├─ train_csnn_single_step.py  # Train/evaluate single-step SNN
│  └─ train_csnn_temporal.py     # Train/evaluate temporal SNN
│
├─ result_scripts/
│  ├─ classification_accuracy.py # Utilities for plotting/analysis of model comparison
│  └─ plot_training_curves.py    # Utilities for plotting/analysis of loss and accuracy
│
├─ results/
│  ├─ cnn_loss.json              # Saved per-epoch training loss for CNN
│  ├─ accuracy_comparison.png    # Example metrics dict (loss/accuracy curves)
│  └─ tbd                        # More data planned to be added
│
├─ data/                         # Fashion-MNIST will be downloaded here
├─ .venv/                        # Project virtual environment (not committed)
├─ requirements.txt
└─ README.md
```

---

## Models

### CNN Baseline
Standard convolutional neural network for 28×28 grayscale Fashion-MNIST
images.

### CSNN Single Step
Spiking CNN using snnTorch LIF layers. Processes each image in a single time step; behaves like a hybrid CNN/SNN model. Used to compare “one-shot” spiking behavior to the CNN baseline.

### CSNN Temporal
Fully temporal spiking CNN that runs over multiple time steps. Uses spike-encoded inputs (e.g., Poisson rate encoding). Output is decoded over time (e.g., rate-based decoding of class logits).

---

## Environment Setup
A virtual Python environment is reccomended. 
To activate a python environment, paste this command into the repo root:
```text
python -m venv .venv
```
---

## Dependencies
Needed dependencies can be installed with the command:
```text
pip install -r requirements.txt
```
---

## Training Scripts
All training scripts are run from the repository root so package imports work properly.
Scripts will download needed dataset (if not previously downloaded), train for the specified amount of epochs and other specified settings, and saves necessary data to results folder.

### Training The CNN:
```text
py -m train_scripts.train_cnn
```

### Training The CSNN Single Step:
```text
py -m train_scripts.train_csnn_single_step
```

### Training The CSNN Temporal:
```text
py -m train_scripts.train_csnn_temporal
```

---

## Testing Results

### Model Accuracy Comparison
| Model        | Epochs | Accuracy % |
|--------------|--------|------------|
| CNN          | 10     | 92.12      |
| CSNN-Single  |        |            |
| CSNN-Temporal|        |            |

Todo: Add comparison graph

### Training Loss Curves
Todo: add training loss curve comparison graph

---

## Reproducability

**Python 3.11.6** interpreter was used to create this project.

### Parameters
Below are the parameters I used on the models to generate my results:

| Parameter | Model: CNN  | Model: CSNN-Single | Model: CNN-Temporal |
|-----------|-------------|--------------------|---------------------|
| Batch Size| 128         |                    |                     |           
| Loss Type | CrossEntropy|                    |                     |
| Optimizer | Adam        |                    |                     |
| Learn Rate| 1e-3        |                    |                     |
| Seed      | 0           |                    |                     |
| Num Steps | N/A         | 1                  |                     |

---

## License
This project is licensed under the terms of the **MIT License**.  
See the [`LICENSE`](./LICENSE) file for details.
