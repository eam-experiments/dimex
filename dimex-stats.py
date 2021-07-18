import random
import numpy as np

# Median of data
cut_point = 7268.5

# Load dictionary with labels as keys (structured array) 
frequencies = np.load('Features/media.npy', allow_pickle=True).item()
for label in frequencies:
    frequencies[label] = 0

# Load DIMEX-100 labels
labels = np.load('Features/feat_Y.npy')
features = np.load('Features/feat_X.npy', allow_pickle=True)
data = [(features[i], labels[i]) for i in range(0, len(features))]
random.shuffle(data)

for label in labels:
    frequencies[label] += 1

# Reduce the number of instances of over-represented classes
pairs = []
for f, l in data:
    if frequencies[l] > cut_point:
        frequencies[l] -= 1
    else:
        pairs.append((f, l))

random.shuffle(pairs)

features = [p[0] for p in pairs]
labels = [p[1] for p in pairs]

np.save('Features/rand_Y.npy', labels)
np.save('Features/rand_X.npy', features)
