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
    def __init__(self, n: int, m: int, tolerance = 0, sigma=0.25, iota=1.0):
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
        sigma:
            The standard deviation of the normal distribution
            used in remembering, as percentage of the number of
            characteristics. Default: None, in which case
            half the number of characteristics is used.
        """
        self._n = n
        self._m = m+1
        self._t = tolerance
        self._absolute_max = 1023
        self._sigma = sigma*m
        self._iota = iota
        self._scale = normpdf(0, 0, self._sigma)

        # It is m+1 to handle partial functions.
        self._relation = np.zeros((self._m, self._n), dtype=np.int)
        self._entropies = np.zeros(self._n, dtype=float)
        self._means = np.zeros(self._n, dtype=float)

        # A flag to know whether entropies and means are up to date.
        self._updated = True

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
    def absolute_max_value(self):
        return self._absolute_max

    @property
    def entropies(self):
        if (not self._updated):
            self._updated = self._update()
        return self._entropies

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        return np.mean(self.entropies)

    @property
    def means(self):
        if (not self._updated):
            self._updated = self._update()
        return self._means

    @property
    def mean(self):
        return np.mean(self.means)

    @property
    def max_value(self):
        return np.max(self.relation)

    @property
    def undefined(self):
        return self.m

    @property
    def sigma(self):
        return self._sigma / self.m
    
    @sigma.setter
    def sigma(self, s):
        self._sigma = abs(s*self.m)
        self._scale = normpdf(0, 0, self._sigma)

    @property
    def iota(self):
        return self._iota

    @iota.setter
    def iota(self, i):
        if (i < 0):
            raise ValueError('Iota must be a non negative number.')
        self._iota = i
         
    def _update(self):
        self._update_entropies()
        self._update_means()
        return True

    def _update_entropies(self):
        totals = self.relation.sum(axis=0)  # sum of cell values by columns
        totals = np.where(totals == 0, 1, totals)
        matrix = self.relation/totals
        matrix = -matrix*np.log2(np.where(matrix == 0.0, 1.0, matrix))
        self._entropies = matrix.sum(axis=0)

    def _update_means(self):
        sums = np.sum(self.relation, axis=0, dtype=float)
        counts = np.count_nonzero(self.relation, axis=0)
        counts = np.where(counts == 0, 1, counts)
        self._means = (sums/counts)/self.max_value

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
        return self._normalize(self.relation[:, j], v, self._sigma, self._scale)

    # Choose a value for feature i.
    def choose(self, j, v):
        if self.is_undefined(v):
            column = self.relation[:,j]
        else:
            column = self._normalize(
                self.relation[:,j], v, self._sigma, self._scale)
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
        return np.mean(weights) / self.max_value

    def abstract(self, r_io) -> None:
        self._relation = np.where(self._relation == self.absolute_max_value, self._relation, self._relation + r_io)

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
        self._updated = False

    def recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        if (not self._updated):
            self._updated = self._update()
        r_io = self.containment(r_io)
        recognized = np.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t
        weight = self._weight(vector)
        recognized = recognized and (self.mean*self._iota <= weight)
        return recognized, weight

    def mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        if (not self._updated):
            self._updated = self._update()
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False)

    def recall(self, vector):
        vector = self.validate(vector)
        if (not self._updated):
            self._updated = self._update()
        accept = self.mismatches(vector) <= self._t
        weight = self._weight(vector)
        accept = accept and (self.mean*self._iota <= weight)
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)
        r_io = self.revalidate(r_io)
        return r_io, accept, weight


class AssociativeMemorySystem:
    def __init__(self, labels: list, n: int, m: int, 
        tolerance = 0, sigma = 0.25, iota = 1.0, kappa = 1.0):
        self._memories = {}
        self.n = n
        self.m = m
        self._kappa = kappa
        self._updated = True
        self._mean = 0.0
        self._labels = labels
        for label in labels:
            self._memories[label] = AssociativeMemory(n, m, tolerance, sigma, iota)

    @property
    def num_mems(self):
        return len(self._memories)

    @property
    def full_undefined(self):
        return np.full(self.n, np.nan)

    @property
    def mean(self):
        if self._updated:
            return self._mean
        self.update()
        return self._mean

    def update(self):
        means = []
        for label in self._memories:
            m = self._memories[label]
            mean = m.mean
            means.append(mean)
        self._mean = np.mean(means)

    def register(self, mem, vector):
        if not (mem in self._memories):
            raise ValueError(f'There is no memory for {mem}')
        self._memories[mem].register(vector)
        self._updated = False

    def recognize(self, vector):
        threshold = self.mean*self._kappa
        for k in self._memories:
            recognized, weight = self._memories[k].recognize(vector, False)
            if recognized and (threshold <= weight):
                return True
        return False

    def recall(self, vector):
        penalty = float('inf')
        threshold = self.mean*self._kappa
        memory = None
        mem_recall = self.full_undefined
        keys = list(self._memories)
        random.shuffle(keys)
        for k in keys:
            recalled, recognized, weight = self._memories[k].recall(vector)
            if recognized and (threshold <= weight):
                entropy = self._memories[k].entropy
                new_penalty = entropy/weight if weight > 0 else float('inf')
                if new_penalty < penalty:
                    penalty = new_penalty
                    memory = k
                    mem_recall = recalled
        return (memory, mem_recall)

