#!/bin/bash

#SBATCH -A csd562
#SBATCH --job-name="pa3"
#SBATCH --output="pa3.data.out"
#SBATCH --partition=compute
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=24
#SBATCH --export=ALL
#SBATCH -t 00:10:00

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

ibrun -np 24 ./apf -n 1800 -i 2000 -x 12 -y 2
ibrun -np 48 ./apf -n 1800 -i 2000 -x 12 -y 4
ibrun -np 96 ./apf -n 1800 -i 2000 -x 8 -y 12
ibrun -np 192 ./apf -n 8000 -i 2000 -x 12 -y 16
ibrun -np 384 ./apf -n 8000 -i 2000 -x 16 -y 24
ibrun -np 480 ./apf -n 8000 -i 2000 -x 24 -y 20

date
echo ">>> Job Ends"
