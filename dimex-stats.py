import random
import numpy as np

# Load dictionary with labels as keys (structured array) 
frequencies = np.load('Features/media.npy', allow_pickle=True).item()
for label in frequencies:
    frequencies[label] = 0

# Load DIMEX-100 labels
labels = np.load('Features/feat_Y.npy')
features = np.load('Features/feat_X.npy', allow_pickle=True)

for label in labels:
    frequencies[label] += 1

# Reduce the number of instances of over-represented classes
average = len(labels)/len(frequencies)
pairs = []
for i in range(len(features)):
    label = labels[i]
    if frequencies[label] > average:
        frequencies[label] -= 1
    else:
        pairs.append((label, features[i]))

random.shuffle(pairs)

for i in range(len(pairs)):
    label, desc = pairs[i]
    labels[i] = label
    features[i] = desc

np.save('Features/rand_Y.npy', labels)
np.save('Features/rand_X.npy', features)