#!/bin/bash
for arg in $( seq 1 70 )
do
	mv inputRGB${arg}.png $(($arg-1)).png
done