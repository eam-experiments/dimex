from random import choice
import constants

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
lines = ['-', '--', '-.', ':']
markers = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8',
           's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_']

fmts = []
for i in constants.all_labels:
    color = choice(colors)
    line = choice(lines)
    marker = choice(markers)
    fmts.append(color+line+marker)
    
print(fmts)
    