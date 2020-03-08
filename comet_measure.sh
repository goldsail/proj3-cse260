#!/bin/bash

#SBATCH -A csd562
#SBATCH --job-name="pa3"
#SBATCH --output="pa3.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 00:03:00

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

N=$1
for ((x=1;x<=N;x++));
do
    if [ $((N % x)) -eq 0 ]
    then
    y=$((N / x))

    ibrun -np $N ./apf -n $2 -i 2000 -x $x -y $y
    ibrun -np $N ./apf -n $2 -i 2000 -x $x -y $y -k
    fi
done

date
echo ">>> Job Ends"
