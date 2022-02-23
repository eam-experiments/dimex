#!/bin/bash

if [ $# != 2 ]; then
    echo "Usage: $0 learned tolerance"
    exit 1
fi
learned=$1
tolerance=$2
for i in 0 1 2 3 4 5 6 7 8; do
    if [ $i == 9 ]; then
        extended="-x"
    else
        extended=""
    fi
    echo "=================== Starting stage ${i}..."
    python eam.py -n $i --learned=$learned --tolerance=$tolerance $extended && \
    python eam.py -f $i --learned=$learned --tolerance=$tolerance $extended && \
    python eam.py -a $i --learned=$learned --tolerance=$tolerance $extended && \
    python eam.py -e $i --learned=$learned --tolerance=$tolerance $extended && \
    python eam.py -i $i --learned=$learned --tolerance=$tolerance $extended && 
    echo "=================== Stage $i finished."
done
