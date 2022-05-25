#! /bin/bash

stages=6
learned=4
tolerance=0
sigma=0.50
iota=0.0
kappa=0.0

for i in "$@"; do
  case $i in
    -g=*|--stages=*)
      stages="${i#*=}"
      shift
      ;;
    -l=*|--learned=*)
      learned="${i#*=}"
      shift
      ;;
    -t=*|--tolerance=*)
      tolerance="${i#*=}"
      shift 
      ;;
    -s=*|--sigma=*)
      sigma="${i#*=}"
      shift 
      ;;
    -i=*|--iota=*)
      iota="${i#*=}"
      shift 
      ;;
    -k=*|--kappa=*)
      kappa="${i#*=}"
      shift 
      ;;
    *)
      echo "Usage $0: [-l | --learned]=learned..."
      exit 1
      ;;
  esac
done

prse=graph_prse_MEAN-english-stg_00
overall=overallgraph_prse_MEAN-english-stg_00
gsize=graph_size_MEAN-english-stg_00
behav=graph_behaviours_MEAN-english-stg_00
recall=recall-graph_prse_MEAN-english-stg_00
totalr=total_recall-graph_prse_MEAN-english-stg_00
suffix=lrn_00${learned}-ext-tol_00${tolerance}.png
img_suffix="-x"
for ((i=0; i<stages;i++)); do
  j=$i
  runs="runs-d${learned}-t${tolerance}-i${iota}-k${kappa}-s${sigma}/stage_$j"
  if [ ! -d $runs ]; then
    echo "Directory $runs does not exist."
    exit 2
  fi
  cd $runs
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

