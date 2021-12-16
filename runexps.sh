#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8; do
    echo "=================== Starting stage ${i}..."
    python eam.py -n $i -m 64 && \
    python eam.py -f $i -m 64 && \
    python eam.py -a $i -m 64 && \
    python eam.py -e $i -m 64 && \
    python eam.py -i $i -m 64 && 
    echo "=================== Stage $i finished."
done
