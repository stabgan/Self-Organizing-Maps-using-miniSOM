# Self Organizing Map
# Credit Card Fraud Detection using MiniSom

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
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
plt.figure(figsize=(10, 10))
plt.bone()
plt.pcolor(som.distance_map().T)
plt.colorbar()

markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plt.plot(w[0] + 0.5,
             w[1] + 0.5,
             markers[y[i]],
             markeredgecolor=colors[y[i]],
             markerfacecolor='None',
             markersize=10,
             markeredgewidth=2)
plt.title('Self-Organizing Map — Credit Card Applications')
plt.show()

# Finding the frauds
# Identify neurons with high mean inter-neuron distance (potential fraud clusters)
mappings = som.win_map(X)

# Collect samples from high-distance neurons dynamically
# instead of hardcoding specific coordinates
distance_map = som.distance_map()
threshold = np.percentile(distance_map, 90)  # top 10% outlier neurons

fraud_candidates = []
for position, samples in mappings.items():
    if distance_map[position] >= threshold:
        fraud_candidates.extend(samples)

if fraud_candidates:
    frauds = np.array(fraud_candidates)
    frauds = sc.inverse_transform(frauds)
    print(f"Detected {len(frauds)} potential fraud applications:")
    print(frauds[:, 0].astype(int))  # Customer IDs
else:
    print("No clear fraud clusters detected with current threshold.")
