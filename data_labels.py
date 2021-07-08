import matplotlib.pyplot as plt
import numpy as np
import dimex

def get_shapes(data):
    shapes = {}
    for d in data:
        if len(d.shape) != 2:
            continue
        if d.shape in shapes:
            shapes[d.shape] += 1
        else:
            shapes[d.shape] = 1
    return shapes

ds = dimex.Sampler()
data, labels = ds.get_data()
print(f'Original data size: {len(data)}')

shapes = get_shapes(data)
common_shape = max(shapes, key=shapes.get)
print(f'Best shape:{common_shape} with {shapes[common_shape]} data')

ndata = []
nlabels = []

for d, l in zip(data, labels):
    if (len(d.shape) == 2) and (d.shape == common_shape):
        e = 10*np.log10(d+1)
        ndata.append(e)
        nlabels.append(l)

print(f'Log10 data size: {len(ndata)}')

np.save('Features/data.npy', ndata)
np.save('Features/labels.npy', nlabels)

print('Done!')



