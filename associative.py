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

import math
import numpy as np
from operator import itemgetter
import random
import time

import constants

def normpdf(x, mean, sd, scale = 1.0):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/(scale*denom)


class AssociativeMemoryError(Exception):
    pass


class AssociativeMemory(object):
    def __init__(self, n: int, m: int, tolerance = 0, zeta=None):
        """
        Parameters
        ----------
        n : int
            The size of the domain (of properties).
        m : int
            The size of the range (of representation).
        tolerance: int
            The number of mismatches allowed between the
            memory content and the cue.
        zeta:
            The standard deviation of the normal distribution
            used in remembering, as percentage of the number of
            characteristics. Default: None, in which case
            half the number of characteristics is used.
        """
        self._n = n
        self._m = m+1
        self._t = tolerance
        self._max = 1023
        percentage = 0.5 if zeta is None else abs(zeta)
        self._zeta = percentage*m
        self._scale = normpdf(0, 0, self._zeta)

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

    @property
    def max_value(self):
        return self._max
        
    @property
    def zeta(self):
        return self._zeta / self.m
    
    @zeta.setter
    def zeta(self, z):
        self._zeta = abs(z*self.n)
        self._scale = normpdf(0, 0, self._zeta)

    def entropies(self):
        """Return the entropy of the Associative Memory."""
        totals = self.relation.sum(axis=0)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = self.relation/totals
        matrix = -matrix*np.log2(np.where(matrix == 0.0, 1.0, matrix))
        return matrix.sum(axis=0)

    def is_undefined(self, value):
        return value == self.undefined

    def vector_to_relation(self, vector):
        relation = np.zeros((self._m, self._n), np.bool)
        relation[vector, range(self.n)] = True
        return relation

    def _normalize(self, column, mean, std, scale):            
        norm = np.array([normpdf(i, mean, std, scale) for i in range(self.m)])
        return norm*column

    def normalized(self, j, v):
        return self._normalize(self.relation[:, j], v, self._zeta, self._scale)

    # Choose a value for feature i.
    def choose(self, j, v):
        if self.is_undefined(v):
            column = self.relation[:,j]
        else:
            column = self._normalize(
                self.relation[:,j], v, self._zeta, self._scale)
        sum = column.sum()
        n = sum*random.random()
        for i in range(self.m):
            if n < column[i]:
                return i
            n -= column[i]
        return self.m - 1
                 
    def _weight(self, vector):
        weights = []
        for i in range(self.n):
            w = 0 if self.is_undefined(vector[i]) \
                else self.relation[vector[i], i]
            weights.append(w)
        return np.mean(weights) / self._max

    def abstract(self, r_io) -> None:
        self._relation = np.where(self._relation == self._max, self._relation, self._relation + r_io)

    def containment(self, r_io):
        return ~r_io[:self.m, :] | self.relation

    # Reduces a relation to a function
    def lreduce(self, vector):
        v = np.array([self.choose(i, vector[i]) for i in range(self.n)])
        return v

    def validate(self, vector):
        """ It asumes vector is an array of floats, and np.nan
            is used to register an undefined value, but it also 
            considerers any negative number or out of range number
            as undefined.
        """
        if len(vector) != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)
        v = np.nan_to_num(vector, copy=True, nan=self.undefined)
        v = np.where((v > self.m) | (v < 0), self.undefined, v)
        return v.astype('int')

    def revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)

    def register(self, vector) -> None:
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)

    def recognize(self, vector, weighted = True):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        recognized = np.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t
        if weighted:
            weight = self._weight(vector)
            return recognized, weight
        else:
            return recognized

    def mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False)

    def recall(self, vector):
        vector = self.validate(vector)
        accept = self.mismatches(vector) <= self._t
        weight = self._weight(vector)
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)
        r_io = self.revalidate(r_io)
        return r_io, accept, weight


class AssociativeMemorySystem:
    def __init__(self, labels: list, n: int, m: int, tolerance = 0):
        self._memories = {}
        self.n = n
        self.m = m
        self.tolerance = tolerance
        self._labels = labels
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
            recognized = self._memories[k].recognize(vector, False)
            if recognized:
                return True
        return False

    def recall(self, vector):
        penalty = float('inf')
        memory = None
        mem_recall = self.full_undefined
        keys = list(self._memories)
        random.shuffle(keys)
        for k in keys:
            recalled, recognized, weight = self._memories[k].recall(vector)
            if recognized:
                entropy = self._memories[k].entropy
                new_penalty = entropy/weight if weight > 0 else float('inf')
                if new_penalty < penalty:
                    penalty = new_penalty
                    memory = k
                    mem_recall = recalled
        return (memory, mem_recall)

