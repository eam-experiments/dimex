#!/bin/bash
########### Script de Ejecución (en Slurm) #########
#SBATCH --partition=gpu
#SBATCH --nodelist=nvd02
#SBATCH --cpus-per-task=22
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

# source ~/.bash_profile
# cd dimex
# conda activate eam
learned=4
tolerance=0
sigma=0.50
iota=0.0
kappa=0.0
extended="-x"
runpath="runs-d$learned-t$tolerance-i$iota-k$kappa-s$sigma"
ok=1
echo "Storing results in $runpath"
for i in 0 1 2 3 4 5 ; do
    j=$i
    if [  $ok != 1 ]; then
        exit 1
    else
	ok=0
    fi
    echo "=================== Starting stage ${i}..."
    python eam.py -n $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -f $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -a $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -e $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -i $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath && \
    python eam.py -r $j --learned=$learned --tolerance=$tolerance $extended --sigma=$sigma --iota=$iota --kappa=$kappa --runpath=$runpath &&
    ok=1
    echo "=================== Stage $i finished."
done

