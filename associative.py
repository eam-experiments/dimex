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

        # it is m+1 to handle partial functions.
        self._relation = np.zeros((self._m, self._n), dtype=np.bool)

    def __str__(self):
        relation = np.zeros((self.m, self.n), dtype=np.unicode)
        relation[:] = 'O'
        r, c = np.nonzero(self.relation)
        for i in zip(r, c):
            relation[i] = 'X'
        return str(relation)

    @property
    def n(self):
        return self._n

    @property
    def m(self):
        return self._m-1

    @property
    def relation(self):
        return self._relation[:self.m,:self.n]

    @property
    def entropy(self) -> float:
        """Return the entropy of the Associative Memory."""
        e = 0.0  # entropy
        v = self.relation.sum(axis=0)  # number of marked cells in the columns
        v = np.where(v == 0, 1, v)
        v = np.log2(v)
        e = v.sum()/self.n
        return e

    @property
    def undefined(self):
        return self.m

    def is_undefined(self, value):
        return value == self.undefined


    def vector_to_relation(self, vector):
        relation = np.zeros((self._m, self._n), np.bool)
        relation[vector, range(self.n)] = True
        return relation


    # Choose a value for feature i.
    def choose(self, i, v):
        if not (self.is_undefined(v) or self.relation[v, i]):
            return self.undefined

        values = np.where(self.relation[:self.m,i])[0]
        if len(values) == 0:
            return self.undefined
        if self.is_undefined(v) or not (v in values):
            return random.choice(values)
        if len(values) == 1:
            return v
        else:
            vj = np.where(values == v)[0][0]
            j = round(random.triangular(0, len(values)-1, vj))
            return values[j]
                 

    def abstract(self, r_io) -> None:
        self._relation = self._relation | r_io


    def containment(self, r_io):
        return ~r_io[:self.m, :self.n] | self.relation


    # Reduces a relation to a function
    def lreduce(self, vector):
        v = [self.choose(i, vector[i]) for i in range(self.n)]
        v = np.array(v)
        return v


    def validate(self, vector):
        # Forces it to be a vector.
        v = np.ravel(vector)

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
        resp_mems = []
        for k in self._memories:
            recalled, recognized = self._memories[k].recall(vector)
            if recognized:
                entropy = self._memories[k].entropy
                record = (k, entropy, recalled)
                resp_mems.append(record)
        if not resp_mems:
            return (None, self.full_undefined)
        else:
            k = None
            recalled = None
            entropy = float('inf')
            for record in resp_mems:
                if record[1] < entropy:
                    k = record[0]
                    entropy = record[1]
                    recalled = record[2]
            return (k, recalled)
        


