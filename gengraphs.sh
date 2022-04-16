#! /bin/bash

learned=4
tolerance=0
sigma=50
iota=1

prse=graph_prse_MEAN-english-stg_00
overall=overallgraph_prse_MEAN-english-stg_00
gsize=graph_size_MEAN-english-stg_00
behav=graph_behaviours_MEAN-english-stg_00
recall=recall-graph_prse_MEAN-english-stg_00
totalr=total_recall-graph_prse_MEAN-english-stg_00
suffix=lrn_00${learned}-ext-tol_00${tolerance}.png
img_suffix="-x"
for i in 0 1 2 3 4 5 6 7 8 9 10; do
    j=$i
    cd runs-32-d${learned}-t${tolerance}-s${sigma}-i${iota}-x/stage_$j
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

