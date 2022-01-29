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

# File originally create by Raul Peralta-Lozada.

import numpy as np
from operator import itemgetter
import random
import time

import constants

class AssociativeMemoryError(Exception):
    pass


class AssociativeMemory(object):
    def __init__(self, n: int, m: int, tolerance = 0):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        """
        self._n = n
        self._m = m+1
        self._t = tolerance
        self._max = 256

        # it is m+1 to handle partial functions.
        self._relation = np.zeros((self._m, self._n), dtype=np.int)

    def __str__(self):
        return str(self.relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

    @property
    def relation(self):
        return self._relation[:self.m,:]

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        entropies = self.entropies()
        return entropies.sum()/self.n

    @property
    def undefined(self):
        return self.m

    def entropies(self):
        """Return the entropy of the Associative Memory."""
        matrix = np.copy(self.relation)
        totals = matrix.sum(axis=0)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = matrix/totals
        matrix = -matrix*np.log2(np.where(matrix == 0.0, 1.0, matrix))
        return matrix.sum(axis=0)

    def is_undefined(self, value):
        return value == self.undefined

    def vector_to_relation(self, vector):
        relation = np.zeros((self._m, self._n), np.bool)
        relation[vector, range(self.n)] = True
        return relation

    # Choose a value for feature i.
    def choose(self, j, v):
        if self.is_undefined(v):
            values = np.where(self.relation[:self.m,j])[0]
            return random.choice(values)
        if not self.relation[v, j]:
            return self.undefined
        bottom = 0
        for i in range(v, -1, -1):
            if not self.relation[i,j]:
                bottom = i + 1
                break
        top = self.m-1
        for i in range(v, self.m):
            if not self.relation[i,j]:
                top = i - 1
                break
        if bottom == top:
            return v
        else:
            sum = self.relation[bottom:top+1, j].sum()
            n = int(sum*random.random())
            for i in range(bottom,top):
                if n < self.relation[i,j]:
                    return i
                n -= self.relation[i,j]
            return top
                 
    def abstract(self, r_io) -> None:
        self._relation = (self._relation + r_io) % self._max

    def containment(self, r_io):
        return ~r_io[:self.m, :] | self.relation


    # Reduces a relation to a function
    def lreduce(self, vector):
        v = np.array([self.choose(i, vector[i]) for i in range(self.n)])
        return v

    def validate(self, vector):
        # Forces it to be a vector.
        v = np.copy(vector)

        if len(v) != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)
        v = np.nan_to_num(v, copy=False, nan=self.undefined)
        v = np.where((v > self.m) | (v < 0), self.undefined, v)
        return v.astype('int')

    def revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)

    def register(self, vector) -> None:
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)

    def recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t

    def mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False)

    def recall(self, vector):
        vector = self.validate(vector)
        accept = self.mismatches(vector) <= self._t
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)
        r_io = self.revalidate(r_io)
        return r_io, accept


class AssociativeMemorySystem:
    def __init__(self, labels: list, n: int, m: int, tolerance = 0):
        self._memories = {}
        self.n = n
        self.m = m
        self.tolerance = tolerance
        for label in labels:
            self._memories[label] = AssociativeMemory(n, m, tolerance)

    @property
    def num_mems(self):
        return len(self._memories)

    @property
    def full_undefined(self):
        return np.full(self.n, np.nan)

    def register(self, mem, vector):
        if not (mem in self._memories):
            raise ValueError(f'There is no memory for {mem}')
        self._memories[mem].register(vector)

    def recognize(self, vector):
        for k in self._memories:
            recognized = self._memories[k].recognize(vector)
            if recognized:
                return True
        return False

    def recall(self, vector):
        entropy = float('inf')
        memory = None
        mem_recall = self.full_undefined
        for k in self._memories:
            recalled, recognized = self._memories[k].recall(vector)
            if recognized:
                new_entropy = self._memories[k].entropy
                if new_entropy < entropy:
                    entropy = new_entropy
                    memory = k
                    mem_recall = recalled
        return (memory, mem_recall)

