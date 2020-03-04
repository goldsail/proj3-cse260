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
#include <vector>

using namespace std;

#define TAG_INIT_E (10)
#define TAG_INIT_R (11)
#define TAG_GATH_SUMS (20)
#define TAG_GATH_LINF (21)

#include "cblock.h"
#include "time.h"
#include "apf.h"
#include <math.h>
#include "Plotting.h"
#include <mpi.h>
extern control_block cb;

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);
double L2Norm(double sumSq);

void printMat(const char mesg[], double *E, int m, int n);
pair<int, int> computeBlockSize(int m, int n, int x, int y);
pair<int, int> computeBlockSize(int m, int n, int x, int y, int rankx, int ranky);
int getRank();
int getNProcs();
int composeRank(int rankx, int ranky, int x, int y);
pair<int, int> decomposeRank(int rank, int x, int y);

void gather(int m, int n, double &sumSq, Plotter *plotter, double &L2, double &Linf) {
  // printf("[INFO] gather result at node %d\n", getRank());
  int M = cb.m;
  int N = cb.n;
  double *ssqs, *maxs;

  vector<MPI_Request> reqs;

  if (getRank() == 0) {
    ssqs = (double*)malloc(sizeof(double) * cb.px * cb.py);
    maxs = (double*)malloc(sizeof(double) * cb.px * cb.py);
    int curX, curY;
    int posX = 0;
    for (int rx = 0; rx < cb.px; rx++, posX += (curX - 1)) {
      int posY = 0;
      for (int ry = 0; ry < cb.py; ry++, posY += (curY - 1)) {
        {
          auto tmp = computeBlockSize(M, N, cb.px, cb.py, rx, ry);
          curX = tmp.first + 2;
          curY = tmp.second + 2;
        }
        int rank = composeRank(rx, ry, cb.px, cb.py);
        MPI_Request req;
        MPI_Datatype bufferTypeBlock;
        MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &bufferTypeBlock); // block buffer
        MPI_Type_commit(&bufferTypeBlock);
        MPI_Irecv(&ssqs[rx*cb.py + ry], 1, bufferTypeBlock, rank, TAG_GATH_SUMS, MPI_COMM_WORLD, &req);
        reqs.push_back(req);
        MPI_Irecv(&maxs[rx*cb.py + ry], 1, bufferTypeBlock, rank, TAG_GATH_LINF, MPI_COMM_WORLD, &req);
        reqs.push_back(req);
      }
    }
  }

  {
    MPI_Request req;
    MPI_Datatype bufferTypeBlock;
    MPI_Type_vector(1, 1, 1, MPI_DOUBLE, &bufferTypeBlock); // block buffer
    MPI_Type_commit(&bufferTypeBlock);
    MPI_Isend(&sumSq, 1, bufferTypeBlock, 0, TAG_GATH_SUMS, MPI_COMM_WORLD, &req);
    reqs.push_back(req);
    MPI_Isend(&Linf, 1, bufferTypeBlock, 0, TAG_GATH_LINF, MPI_COMM_WORLD, &req);
    reqs.push_back(req);
  }

  vector<MPI_Status> status(reqs.size());
  MPI_Waitall(reqs.size(), reqs.data(), status.data());
  MPI_Barrier(MPI_COMM_WORLD);
  //  printMat2("Rank 0 Matrix E_prev", E_prev, m,n);  // return the L2 and infinity norms via in-out parameters

  if (getRank() == 0) {
    sumSq = 0.0;
    for (int rx = 0; rx < cb.px; rx++) {
      for (int ry = 0; ry < cb.py; ry++) {
        Linf = max(Linf, fabs(maxs[rx*cb.py + ry]));
        sumSq += ssqs[rx*cb.py + ry];
      }
    }
    L2 = L2Norm(sumSq);
  }
}

static void fill(double *E_prev, double *R, int m, int n) {
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

//
// Initialization
//
// We set the right half-plane of E_prev to 1.0, the left half plane to 0
// We set the botthom half-plane of R to 1.0, the top half plane to 0
// These coordinates are in world (global) coordinate and must
// be mapped to appropriate local indices when parallelizing the code
//
void init (double *E,double *E_prev,double *R,int m,int n){
    const int M = m, N = n; // global size
    {
      auto tmp = computeBlockSize(M, N, cb.px, cb.py);
      m = tmp.first; // local size
      n = tmp.second; // local size
    }

    // node 0 initializes and sends to all nodes including itself
    if (getRank() == 0) {
      double *gE_prev = (double*)malloc(sizeof(double) * (M+2) * (N+2)); // global initial condition
      double *gR = (double*)malloc(sizeof(double) * (M+2) * (N+2)); // global initial condition
      fill(gE_prev, gR, M, N);

      // send initial condition to other nodes
      int curX, curY;
      int posX = 0;

      for (int rx = 0; rx < cb.px; rx++, posX += (curX - 2)) {
        int posY = 0;
        for (int ry = 0; ry < cb.py; ry++, posY += (curY - 2)) {
          {
            auto tmp = computeBlockSize(M, N, cb.px, cb.py, rx, ry);
            curX = tmp.first + 2;
            curY = tmp.second + 2;
          }
          int rank = composeRank(rx, ry, cb.px, cb.py);
          MPI_Request req;
          MPI_Datatype bufferTypeBlock;
          MPI_Type_vector(curX, curY, N+2, MPI_DOUBLE, &bufferTypeBlock); // block buffer
          MPI_Type_commit(&bufferTypeBlock);
          MPI_Isend(&gE_prev[posX*(N+2) + posY], 1, bufferTypeBlock, rank, TAG_INIT_E, MPI_COMM_WORLD, &req);
          MPI_Isend(&gR[posX*(N+2) + posY], 1, bufferTypeBlock, rank, TAG_INIT_R, MPI_COMM_WORLD, &req);
        }
      }
    }

    // all nodes including node 0 receive from node 0
    {
      MPI_Request reqs[2];
      MPI_Status stats[2];
      MPI_Datatype bufferTypeBlock;
      MPI_Type_vector(m+2, n+2, n+2, MPI_DOUBLE, &bufferTypeBlock); // block buffer
      MPI_Type_commit(&bufferTypeBlock);
      MPI_Irecv(&E_prev[0], 1, bufferTypeBlock, 0, TAG_INIT_E, MPI_COMM_WORLD, &reqs[0]);
      MPI_Irecv(&R[0], 1, bufferTypeBlock, 0, TAG_INIT_R, MPI_COMM_WORLD, &reqs[1]);

      MPI_Waitall(2, reqs, stats);
    }

    MPI_Barrier(MPI_COMM_WORLD);

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

pair<int, int> computeBlockSize(int m, int n, int x, int y) {
  auto temp = decomposeRank(getRank(), x, y);
  int rankx = temp.first;
  int ranky = temp.second;
  return computeBlockSize(m, n, x, y, rankx, ranky);
}

pair<int, int> computeBlockSize(int m, int n, int x, int y, int rankx, int ranky) {
  int a = m/x + (rankx < m%x);
  int b = n/y + (ranky < n%y);
  return make_pair(a, b); // a and b do not inlcude ghost cells!
}

int composeRank(int rankx, int ranky, int x, int y) {
  return rankx * y + ranky;
}

pair<int, int> decomposeRank(int rank, int x, int y) {
  return make_pair(rank / y, rank % y);
}

double *alloc1D(int m,int n){
    auto pair = computeBlockSize(m-2, n-2, cb.px, cb.py);
    int nx = pair.first + 2;
    int ny = pair.second + 2;
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
