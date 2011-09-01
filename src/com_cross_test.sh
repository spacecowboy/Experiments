#!/bin/bash

for nodes in {1..10}
do
	file="com_cross_"$nodes"node.txt"
	echo "nice python cox_genetic_committee_cross.py "$nodes" > "$file
	nice python cox_genetic_committee_cross.py $nodes > $file
done
