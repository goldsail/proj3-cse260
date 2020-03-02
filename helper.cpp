/*
 * Utilities for the Aliev-Panfilov code
 * Scott B. Baden, UCSD
 * Nov 2, 2015
 */

#include <iostream>
#include <assert.h>
// Needed for memalign
#include <malloc.h>
#include <utility>

using namespace std;

#include "cblock.h"
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include <mpi.h>
extern control_block cb;

void printMat(const char mesg[], double *E, int m, int n);
pair<pair<int, int>, pair<int, int> > computeBlockSize(int m, int n, int x, int y);



//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    {
      auto tmp = computeBlockSize(m, n, cb.px, cb.py);
      m = tmp.first.first;
      n = tmp.first.second;
    }

    int i;

    for (i=0; i < (m+2)*(n+2); i++)
        E_prev[i] = R[i] = 0;

    for (i = (n+2); i < (m+1)*(n+2); i++) {
      	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

              // Need to compute (n+1)/2 rather than n/2 to work with odd numbers
      	if(colIndex == 0 || colIndex == (n+1) || colIndex < ((n+1)/2+1))
      	    continue;

              E_prev[i] = 1.0;
    }

    for (i = 0; i < (m+2)*(n+2); i++) {
      	int rowIndex = i / (n+2);		// gives the current row number in 2D array representation
      	int colIndex = i % (n+2);		// gives the base index (first row's) of the current index

              // Need to compute (m+1)/2 rather than m/2 to work with odd numbers
      	if(colIndex == 0 || colIndex == (n+1) || rowIndex < ((m+1)/2+1))
      	    continue;

              R[i] = 1.0;
    }
    // We only print the meshes if they are small enough
#if 1
    printMat("E_prev",E_prev,m,n);
    printMat("R",R,m,n);
#endif
}

int getRank() {
  int myrank=0;
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  return myrank;
}

int getNProcs() {
    int nprocs=1;
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    return nprocs;
}

pair<pair<int, int>, pair<int, int>> computeBlockSize(int m, int n, int x, int y) {
  int rank = getRank(), rankx = rank/y, ranky = rank%y;
  int a = m/x + (rankx < m%x);
  int b = n/y + (ranky < n%y);
  return make_pair(make_pair(a, b), make_pair(rankx, ranky)); // a and b do not inlcude ghost cells!
}

int composeRank(int rankx, int ranky, int x, int y) {
  return rankx * y + ranky;
}

double *alloc1D(int m,int n){
    auto pair = computeBlockSize(m-2, n-2, cb.px, cb.py);
    int nx=pair.first.first+2, ny=pair.first.second+2;
    double *E;
    // Ensures that allocatdd memory is aligned on a 16 byte boundary
    assert(E= (double*) memalign(16, sizeof(double)*nx*ny) ); // TODO: change alignment size
    return(E);
}

void printMat(const char mesg[], double *E, int m, int n){
    int i;
#if 0
    if (m>8)
      return;
#else
    if (m>34)
      return;
#endif
    printf("%s\n",mesg);
    for (i=0; i < (m+2)*(n+2); i++){
       int rowIndex = i / (n+2);
       int colIndex = i % (n+2);
       if ((colIndex>0) && (colIndex<n+1))
          if ((rowIndex > 0) && (rowIndex < m+1))
            printf("%6.3f ", E[i]);
       if (colIndex == n+1)
	    printf("\n");
    }
}
