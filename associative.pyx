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
cimport cython
import random
import time
import constants

cdef class AssociativeMemoryError(Exception):
    pass


cdef class AssociativeMemory:
    
    def __init__(self, int n, int m, int tolerance = 0):
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
        self._relation = np.zeros((self._m, self._n), dtype=np.int)

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
    def entropy(self):
        """Return the entropy of the Associative Memory."""
        cdef float e = 0.0  # entropy
        v = self.relation.sum(axis=0)  # number of marked cells in the columns
        for vi in v:
            if vi != 0:
                e -= np.log2(vi)
        e *= (-1.0 / self.n)
        return e

    @property
    def undefined(self):
        return self.m

    cpdef is_undefined(self, int value):
        return value == self.undefined


    cpdef vector_to_relation(self, vector):
        relation = np.zeros((self._m, self._n), np.bool)
        relation[vector, range(self.n)] = True
        return relation


    # Choose a value for feature i.
    cpdef choose(self, int i, int v):
        if not (self.is_undefined(v) or self.relation[v, i]):
            return self.undefined

        values = np.where(self.relation[:self.m,i])[0]
        if len(values) == 0:
            return self.undefined
        if self.is_undefined(v):
            return random.choice(values)
        else:
            vj = np.where(values == v)[0][0]
            j = round(random.triangular(0, len(values)-1, vj))
            return values[j]
                 

    cpdef void abstract(self, r_io):
        self._relation = self._relation | r_io


    cpdef containment(self, r_io):
        return ~r_io[:self.m, :self.n] | self.relation


    # Reduces a relation to a function
    cpdef lreduce(self, vector):
        v = np.full(self.n, self.undefined)
        for i in range(self.n):
            v[i] = self.choose(i, vector[i])
        return v


    cpdef validate(self, vector):
        # Forces it to be a vector.
        v = np.ravel(vector)

        if len(v) != self.n:
            raise ValueError('Invalid size of the input data. Expected', self.n, 'and given', vector.size)
        v = np.nan_to_num(v, copy=False, nan=self.undefined)
        v = np.where((v > self.m) | (v < 0), self.undefined, v)
        return v.astype('int')
        

    cpdef revalidate(self, vector):
        v = vector.astype('float')
        return np.where(v == float(self.undefined), np.nan, v)


    cpdef void register(self, vector):
        vector = self.validate(vector)

        r_io = self.vector_to_relation(vector)
        self.abstract(r_io)


    cpdef recognize(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False) <= self._t


    cpdef mismatches(self, vector):
        vector = self.validate(vector)
        r_io = self.vector_to_relation(vector)
        r_io = self.containment(r_io)
        return np.count_nonzero(r_io[:self.m,:self.n] == False)


    cpdef recall(self, vector):
        vector = self.validate(vector)
        accept = self.mismatches(vector) <= self._t
        if accept:
            r_io = self.lreduce(vector)
        else:
            r_io = np.full(self.n, self.undefined)
        r_io = self.revalidate(r_io)
        return r_io, accept


cdef class AssociativeMemorySystem:
    def __init__(self, list labels, int n, int m, int tolerance = 0):
        self._memories = {}
        self.n = n
        self.m = m
        self.tolerance = tolerance
        for label in labels:
            self._memories[label] = AssociativeMemory(n, m, tolerance)

    @property
    def n(self):
        return self.n

    @property
    def m(self):
        return self.m

    @property
    def num_mems(self):
        return len(self._memories)

    @property
    def full_undefined(self):
        return np.full(self.n, np.nan)

    cpdef void register(self, int mem, vector):
        if not (mem in self._memories):
            raise ValueError(f'There is no memory for {mem}')
        self._memories[mem].register(vector)

    cpdef recognize(self, vector):
        for k in self._memories:
            recognized = self._memories[k].recognize(vector)
            if recognized:
                return True
        return False

    cpdef recall(self, vector):
        cdef list resp_mems = []
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
        


