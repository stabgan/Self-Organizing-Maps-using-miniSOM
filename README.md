# 🧠 Self-Organizing Maps — Credit Card Fraud Detection

A Self-Organizing Map (SOM) implementation using **MiniSom** to detect potential fraud in credit card applications. The model learns topological patterns in high-dimensional customer data and surfaces outlier neurons that may correspond to fraudulent behavior.

---

## 📖 Methodology

1. **Data ingestion** — Load the `Credit_Card_Applications.csv` dataset containing 15 features per applicant.
2. **Feature scaling** — Normalize all features to `[0, 1]` using `MinMaxScaler`.
3. **SOM training** — A 10×10 grid of neurons is trained over 100 iterations. Each neuron competes to represent input vectors; weights are updated via a Gaussian neighborhood function with decaying learning rate and sigma.
4. **Visualization** — The SOM distance map (U-Matrix) is plotted. High-distance neurons indicate regions where the map topology stretches — potential anomaly zones. Approved vs. rejected applicants are overlaid with distinct markers.
5. **Fraud detection** — Neurons whose mean inter-neuron distance exceeds the 90th percentile are flagged. All customer samples mapped to those neurons are reported as fraud candidates.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| 🐍 Language | Python 3.8+ |
| 🧮 Numerics | NumPy |
| 📊 Visualization | Matplotlib |
| 📁 Data handling | Pandas |
| ⚙️ Preprocessing | scikit-learn (`MinMaxScaler`) |
| 🧠 SOM engine | MiniSom (bundled) |

---

## 📦 Dependencies

```
numpy
pandas
matplotlib
scikit-learn
```

Install everything at once:

```bash
pip install numpy pandas matplotlib scikit-learn
```

> **Note:** MiniSom is included directly in the repo (`minisom.py`), no separate install needed.

---

## 🚀 How to Run

```bash
cd Self_Organizing_Maps
python som.py
```

The script will:
- Train the SOM on the credit card dataset
- Display the U-Matrix visualization with applicant overlays
- Print detected fraud candidate IDs to the console

---

## ⚠️ Known Issues

- **Hardcoded grid size** — The 10×10 SOM grid and `input_len=15` are tuned for this specific dataset. Different datasets require reconfiguration.
- **Threshold sensitivity** — The 90th-percentile cutoff for fraud detection is a heuristic. Adjust based on domain requirements and false-positive tolerance.
- **Bundled MiniSom** — The included `minisom.py` is an older snapshot. For production use, consider installing the maintained package via `pip install minisom`.
- **No train/test split** — The entire dataset is used for both training and fraud detection (unsupervised), so there is no held-out evaluation.

---

## 📄 License

MIT — see [LICENSE](LICENSE).
