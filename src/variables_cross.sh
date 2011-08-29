#!/bin/bash

source /home/gibson/jonask/virtual_pythons/ann/bin/activate
cd /home/gibson/jonask/Projects/Experiments/src/
echo $1

if [ $#>0 ]
then
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -2 -1 > Plots/netsize/crossvalidation/variables/$1/-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -3 -1 > Plots/netsize/crossvalidation/variables/$1/-3-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -3 -2 > Plots/netsize/crossvalidation/variables/$1/-3-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -3 -2 -1 > Plots/netsize/crossvalidation/variables/$1/-3-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -1 > Plots/netsize/crossvalidation/variables/$1/-4-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -2 > Plots/netsize/crossvalidation/variables/$1/-4-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -2 -1 > Plots/netsize/crossvalidation/variables/$1/-4-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -3 > Plots/netsize/crossvalidation/variables/$1/-4-3.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -3 -1 > Plots/netsize/crossvalidation/variables/$1/-4-3-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -3 -2 > Plots/netsize/crossvalidation/variables/$1/-4-3-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 -4 -3 -2 -1 > Plots/netsize/crossvalidation/variables/$1/-4-3-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -2 > Plots/netsize/crossvalidation/variables/$1/2-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -2 -1 > Plots/netsize/crossvalidation/variables/$1/2-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -3 > Plots/netsize/crossvalidation/variables/$1/2-3.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -3 -1 > Plots/netsize/crossvalidation/variables/$1/2-3-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -3 -2 > Plots/netsize/crossvalidation/variables/$1/2-3-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -3 -2 -1 > Plots/netsize/crossvalidation/variables/$1/2-3-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 > Plots/netsize/crossvalidation/variables/$1/2-4.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -2 > Plots/netsize/crossvalidation/variables/$1/2-4-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -2 -1 > Plots/netsize/crossvalidation/variables/$1/2-4-2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -3 > Plots/netsize/crossvalidation/variables/$1/2-4-3.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -3 -1 > Plots/netsize/crossvalidation/variables/$1/2-4-3-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -3 -2 > Plots/netsize/crossvalidation/variables/$1/2-4-3-2.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -1 > Plots/netsize/crossvalidation/variables/$1/2-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -1 > Plots/netsize/crossvalidation/variables/$1/2-4-1.txt
	echo "Working..."
	nice -n 19 python cox_genetic.py $1 2 -4 -3 -2 -1 > Plots/netsize/crossvalidation/variables/$1/2-4-3-2-1.txt
fi
