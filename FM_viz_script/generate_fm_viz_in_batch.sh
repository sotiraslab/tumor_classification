#!/bin/bash

mkdir fm_pngs

for d in MW*;do
	cd $d
	echo $d
	cp ../fm*.py ./
	echo "Upsampling.."
	python fm_upsample_range_PC.py
	echo "Generating grid plot.."
	python fm_viz_range_PC.py
	echo "Generating single plot with scan.."
	python fm_viz_single_with_scan_PC.py
	rm -rf fm*.py
	dname="$(echo "$d" | cut -d'/' -f1)"
	mv fm_layer44+scan.png ../fm_pngs/"$dname"_fm_layer44+scan.png
	mv fm_mosaic_layer44.png ../fm_pngs/"$dname"_fm_mosaic_layer44.png
	cd ../
done
