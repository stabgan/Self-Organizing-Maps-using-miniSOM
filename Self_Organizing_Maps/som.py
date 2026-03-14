# Self Organizing Map
# Detects potential fraud in credit card applications using MiniSom.

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from minisom import MiniSom


def main():
    # Importing the dataset (resolve path relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "Credit_Card_Applications.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # Feature Scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    X = sc.fit_transform(X)

    # Training the SOM
    som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
    som.random_weights_init(X)
    som.train_random(data=X, num_iteration=100)

    # Visualizing the results
    plt.figure(figsize=(10, 8))
    plt.pcolor(som.distance_map().T, cmap="bone")
    plt.colorbar(label="Mean Inter-Neuron Distance")
    markers = ["o", "s"]
    colors = ["r", "g"]
    for i, x in enumerate(X):
        w = som.winner(x)
        plt.plot(
            w[0] + 0.5,
            w[1] + 0.5,
            markers[y[i]],
            markeredgecolor=colors[y[i]],
            markerfacecolor="None",
            markersize=10,
            markeredgewidth=2,
        )
    plt.title("Self-Organizing Map — Credit Card Applications")
    plt.show()

    # Finding the frauds dynamically
    # Identify outlier neurons whose mean distance exceeds a threshold
    distance_map = som.distance_map()
    threshold = distance_map.mean() + distance_map.std()
    mappings = som.win_map(X)

    fraud_candidates = []
    for (i, j), samples in mappings.items():
        if distance_map[i, j] >= threshold:
            fraud_candidates.extend(samples)

    if fraud_candidates:
        frauds = np.array(fraud_candidates)
        frauds = sc.inverse_transform(frauds)
        print(f"Detected {len(frauds)} potential fraud application(s):")
        print(frauds)
    else:
        print("No outlier neurons found above the threshold.")


if __name__ == "__main__":
    main()
