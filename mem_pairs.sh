#!/usr/bin/bash

prefix="mem_confrix-lbl_"
suffix="-stg_000-lrn_004-ext-tol_000.png"
suffixfive="-stg_000-lrn_004-ext-tol_002.png"
inone="runs-d4-t0-i0.0-k0.0-s0.50"
intwo="runs-d4-t0-i0.1-k0.0-s0.50"
inthree="runs-d4-t0-i0.0-k1.5-s0.50"
infour="runs-d4-t0-i0.1-k1.5-s0.10"
infive="runs-d4-t2-i0.1-k1.5-s0.10"
outdir="runs"

for i in $(seq -f "%03g" 0 21); do
  filename="$prefix$i$suffix"
  filenamefive="$prefix$i$suffixfive"
  echo $filename
  convert "$inone/$filename" "$intwo/$filename" "$inthree/$filename" "$infour/$filename" "$infive/$filenamefive" +append $outdir/$filename
done
