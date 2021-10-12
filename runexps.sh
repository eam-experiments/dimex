#!/bin/bash
for i in 0 1 2 3 4 5 6 7 8; do
    echo "=================== Starting stage ${i}..."
    python eam.py -n $i && \
    python eam.py -f $i && \
    python eam.py -a $i && \
    python eam.py -e $i && \
    python eam.py -i $i && 
    echo "=================== Stage $i finished."
done
