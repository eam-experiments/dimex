#! /bin/bash

learned=4
tolerance=0
sigma=0.10
iota=0.3
kappa=1.5

prse=graph_prse_MEAN-english-stg_00
overall=overallgraph_prse_MEAN-english-stg_00
gsize=graph_size_MEAN-english-stg_00
behav=graph_behaviours_MEAN-english-stg_00
recall=recall-graph_prse_MEAN-english-stg_00
totalr=total_recall-graph_prse_MEAN-english-stg_00
suffix=lrn_00${learned}-ext-tol_00${tolerance}.png
img_suffix="-x"
for i in 0 1 2 3 4 5 6 7 8 9 ; do
    j=$i
    cd runs-d${learned}-t${tolerance}-i${iota}-k${kappa}-s${sigma}/stage_$j
	for f in *.svg; do
		g=`basename $f .svg`
		rsvg-convert -f png -o ${g}.png $f
	done

	convert $prse${j}-$suffix $overall${j}-$suffix +append row1.png
	convert $gsize${j}-$suffix $behav${j}-$suffix +append row2.png
	convert $recall${j}-$suffix $totalr${j}-$suffix +append row3.png
	convert row1.png row2.png row3.png -append stage_${j}${img_suffix}.png
        cp stage_${j}${img_suffix}.png ..
	cd ../..
done

