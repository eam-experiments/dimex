# Copyright [2020] Luis Alberto Pineda Cortés, Rafael Morales Gamboa.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    