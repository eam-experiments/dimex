#! /bin/sh

prse=graph_prse_MEAN-english-stg_00
overall=overallgraph_prse_MEAN-english-stg_00
gsize=graph_size_MEAN-english-stg_00
behav=graph_behaviours_MEAN-english-stg_00
recall=recall-graph_prse_MEAN-english-stg_00
totalr=total_recall-graph_prse_MEAN-english-stg_00
suffix=lrn_000-tol_000.png

for i in 0 1 2 3 4 5 6 7 8; do
    cd runs/stage_$i
	for f in *.svg; do
		g=`basename $f .svg`
		rsvg-convert -f png -o ${g}.png $f
	done

	convert $prse${i}-$suffix $overall${i}-$suffix +append row1.png
	convert $gsize${i}-$suffix $behav${i}-$suffix +append row2.png
	convert $recall${i}-$suffix $totalr${i}-$suffix +append row3.png
	convert row1.png row2.png row3.png -append stage_${i}.png
	rm row?.png
        cp stage_${i}.png ..
	cd ../..
done

