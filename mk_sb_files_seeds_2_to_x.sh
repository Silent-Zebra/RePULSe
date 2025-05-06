#!/bin/bash
# First arg is how many seeds, second arg should be the full filename of the sb file
for i in `seq 2 $1`
do 
	cp $2 $2_s$i
	sed -i "s@seed 1@seed $i@" $2_s$i
	sed -i "s@-J s1@-J s$i@" $2_s$i
	sed -i "s@1.txt@$i.txt@" $2_s$i
	sed -i "s@master_port \(....\)1@master_port \1$i@" $2_s$i
done
