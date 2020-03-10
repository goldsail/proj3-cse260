#!/bin/bash

#SBATCH -A csd562
#SBATCH --job-name="pa3"
#SBATCH --output="q2d_no-blk.o%j"
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 00:10:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yuh023@ucsd.edu

#This job runs with 2  nodes, 24 cores per node for a total of 48 cores.

echo
echo " *** Current working directory"
pwd
echo
echo " *** Compiler"
# Output which  compiler are we using and the environment
mpicc -v
echo
echo " *** Environment"
printenv

echo

echo ">>> Job Starts"
date

for N in 24 48 96
do
	for ((x=1;x<=N;x++));
	do
	    if [ $((N % x)) -eq 0 ]
	    then
	    y=$((N / x))

	    ibrun -np $N ./apf -n 1800 -i 2000 -x $x -y $y
	    ibrun -np $N ./apf -n 1800 -i 2000 -x $x -y $y -k
	    fi
	done
done

date
echo ">>> Job Ends"
