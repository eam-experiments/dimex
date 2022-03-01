# Copyright [2020] Luis Alberto Pineda Cortés, Gibrán Fuentes Pineda,
# Rafael Morales Gamboa.
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

import numpy as np
from associative import *

# Memory for 4 features of size 3
m = AssociativeMemory(5,5)

v0 = np.full(5, 0, dtype=int)
v1 = np.full(5, 1, dtype=int)
v2 = np.full(5, 2, dtype=int)
v3 = np.full(5, 3, dtype=int)
v4 = np.full(5, 4, dtype=int)

vd = np.array([0, 1, 2, 3, 4])

# 0: 0 0 0 1
# 1: 1 0 1 0
# 2: 0 1 0 0

vi = np.array([4, 3, 2, 1, 0])

def testing_recognize():
    m = AssociativeMemory(5,5)
    m.register(vd)
    m.register(vi)
    m.register(v2)
    vs = [v0, v1, v2, vi, vd]
    n = 500*1000
    start = time.perf_counter()
    for _ in range(n):
        for v in vs:
            m.recognize(v)
    end = time.perf_counter()
    return end - start

m.register(v0)
m.register(v1)
m.register(v4)
for _ in range(10):
    m.register(vd)
    m.register(vi)

ams = AssociativeMemorySystem(list(range(3)), 5, 5)
ams.register(0, v0)
ams.register(1, v1)
ams.register(2, v2)
for i in range(3):
    ams.register(i, vd)
    ams.register(i, vi)
    ams.register(i, v2)
for i in range(3):
    ams.register(2, v2)





