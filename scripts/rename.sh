#!/bin/bash
for arg in $( seq 0 9 )
do
	mv image${arg}.png ${arg}.png
done