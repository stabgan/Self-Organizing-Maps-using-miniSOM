# Self-Organizing Maps — Credit Card Fraud Detection

Unsupervised fraud detection on credit card applications using a Self-Organizing Map (SOM) built with [MiniSom](https://github.com/JustGlowing/minisom).

## What It Does

A 10×10 SOM is trained on scaled credit card application features. The distance map highlights outlier neurons — regions where the map's weight vectors differ significantly from their neighbours. Applications mapped to those neurons are flagged as potential fraud.

### Methodology

1. Load and min-max scale the 15 application features to [0, 1].
2. Train a 10×10 SOM for 100 iterations (Gaussian neighbourhood, σ = 1.0, lr = 0.5).
3. Visualise the U-matrix (mean inter-neuron distance) with approved/rejected markers.
4. Dynamically identify outlier neurons (distance > mean + 1 std) and extract mapped applications.

## Dataset

`Credit_Card_Applications.csv` — 690 rows, 15 features + 1 approval label. The label is used only for visualisation markers, not for training (unsupervised).

## 🛠 Tech Stack

| Tool | Purpose |
|------|---------|
| 🐍 Python 3 | Runtime |
| 📊 MiniSom | Self-Organizing Map implementation |
| 🔢 NumPy | Numerical operations |
| 📈 Matplotlib | Visualisation |
| 🐼 Pandas | Data loading |
| ⚙️ scikit-learn | Feature scaling (`MinMaxScaler`) |

## Installation

```bash
pip install numpy matplotlib pandas scikit-learn
```

> `minisom.py` is bundled in the repo — no separate install needed.

## Usage

```bash
cd Self_Organizing_Maps
python som.py
```

A colour-mapped SOM visualisation will appear. Detected fraud candidates are printed to the console.

## ⚠️ Known Issues

- The bundled `minisom.py` is a legacy snapshot. For production use, install the latest version via `pip install minisom`.
- Fraud threshold (mean + 1 std) is a heuristic — tune it for your data.
