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

# cython: infer_types=True
cimport numpy as np
cimport cython

ctypedef fused integer:
    int

ctypedef fused int_double:
    int
    double

cdef class AssociativeMemoryError(Exception):
    pass


cdef class AssociativeMemory:
    cdef int _n
    cdef int _m
    cdef int _t
    cdef np.int_t[:,:] _relation

    cpdef is_undefined(self, int value)

    cpdef vector_to_relation(self, vector)

    # Choose a value for feature i.
    cpdef choose(self, int i, int v)

    cpdef void abstract(self, r_io)

    cpdef containment(self, r_io)

    # Reduces a relation to a function
    cpdef lreduce(self, vector)

    cpdef validate(self, vector)

    cpdef revalidate(self, vector)


    cpdef void register(self, vector)

    cpdef recognize(self, vector)

    cpdef mismatches(self, vector)

    cpdef recall(self, vector)


cdef class AssociativeMemorySystem:

    cdef dict memories
    cdef int n
    cdef int m
    cdef int tolerance

    cpdef void register(self, int mem, vector)

    cpdef recognize(self, vector)

    cpdef recall(self, vector)
