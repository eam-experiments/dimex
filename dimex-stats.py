import numpy as np

# Load dictionary with labels as keys (structured array) 
frequencies = np.load('Features/media.npy', allow_pickle=True).item()

for label in frequencies:
    frequencies[label] = 0

# Load DIMEX-100 labels
labels = np.load('Features/feat_Y.npy')
for label in labels:
    frequencies[label] += 1

print(frequencies)
