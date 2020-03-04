/*
 * Solves the Aliev-Panfilov model  using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory
 *
 * Modified and  restructured by Scott B. Baden, UCSD
 *
 */

#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include "time.h"
#include "apf.h"
#include "Plotting.h"
#include "cblock.h"
#include <emmintrin.h>

#include <mpi.h>

using namespace std;

#define TAG_COMP_E (0)
#define TAG_COMP_EP (1)
#define TAG_COMP_R (2)

void repNorms(double l2norm, double mx, double dt, int m,int n, int niter, int stats_freq);
void stats(double *E, int m, int n, double *_mx, double *sumSq);
void printMat2(const char mesg[], double *E, int m, int n);

pair<int, int> computeBlockSize(int m, int n, int x, int y);
pair<int, int> computeBlockSize(int m, int n, int x, int y, int rankx, int ranky);
int getRank();
int getNProcs();
int composeRank(int rankx, int ranky, int x, int y);
pair<int, int> decomposeRank(int rank, int x, int y);

void gather(int m, int n, double &sumSq, Plotter *plotter, double &L2, double &Linf);

extern control_block cb;

// #ifdef SSE_VEC
// If you intend to vectorize using SSE instructions, you must
// disable the compiler's auto-vectorizer
// __attribute__((optimize("no-tree-vectorize")))
// #endif

// The L2 norm of an array is computed by taking sum of the squares
// of each element, normalizing by dividing by the number of points
// and then taking the sequare root of the result
//
double L2Norm(double sumSq){
    double l2norm = sumSq /  (double) ((cb.m)*(cb.n));
    l2norm = sqrt(l2norm);
    return l2norm;
}

void solve(double **_E, double **_E_prev, double *R, double alpha, double dt, Plotter *plotter, double &L2, double &Linf){

 // Simulated time is different from the integer timestep number
 double t = 0.0;

 double *E = *_E, *E_prev = *_E_prev;
 double *R_tmp = R;
 double *E_tmp = *_E;
 double *E_prev_tmp = *_E_prev;
 double mx, sumSq;
 int niter;
  const auto blockSize = computeBlockSize(cb.m, cb.n, cb.px, cb.py);
  const int rank = getRank();
  const auto blockRank = decomposeRank(rank, cb.px, cb.py);
  const int rankx = blockRank.first, ranky = blockRank.second;
 const int m = blockSize.first, n = blockSize.second;
 int innerBlockRowStartIndex = (n+2)+1;
 int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  MPI_Request req;

  MPI_Datatype bufferTypeRow;
  MPI_Datatype bufferTypeColumn;
  MPI_Type_vector(n, 1, 1, MPI_DOUBLE, &bufferTypeRow); // row buffer
  MPI_Type_vector(m, 1, n + 2, MPI_DOUBLE, &bufferTypeColumn); // column buffer
  MPI_Type_commit(&bufferTypeRow);
  MPI_Type_commit(&bufferTypeColumn);

// #define DISABLE_COMMUNICATION

 // We continue to sweep over the mesh until the simulation has reached
 // the desired number of iterations
  for (niter = 0; niter < cb.niters; niter++){

      if  (cb.debug && (niter==0)){
	  stats(E_prev,m,n,&mx,&sumSq);
          double l2norm = L2Norm(sumSq);
	  repNorms(l2norm,mx,dt,m,n,-1, cb.stats_freq);
	  if (cb.plot_freq)
	      plotter->updatePlot(E,  -1, m+1, n+1);
      }

   /*
    * Copy data from boundary of the computational box to the
    * padding region, set up for differencing computational box's boundary
    *
    * These are physical boundary conditions, and are not to be confused
    * with ghost cells that we would use in an MPI implementation
    *
    * The reason why we copy boundary conditions is to avoid
    * computing single sided differences at the boundaries
    * which increase the running time of solve()
    *
    */

    // 4 FOR LOOPS set up the padding needed for the boundary conditions
    int i,j;
    vector<MPI_Request> reqs;

    // Fills in the TOP Ghost Cells
    if (rankx == 0) {
      for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
      }
    } else {
      // communicate
#ifndef DISABLE_COMMUNICATION
      MPI_Isend(&E[1 + (n+2)*1], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&E_prev[1 + (n+2)*1], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&R[1 + (n+2)*1], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E[1 + (n+2)*0], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E_prev[1 + (n+2)*0], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&R[1 + (n+2)*0], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
#endif
    }



    // Fills in the RIGHT Ghost Cells
    if (ranky + 1 == cb.py) {
      for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
      }
    } else {
      // communicate
#ifndef DISABLE_COMMUNICATION
      MPI_Isend(&E[n + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&E_prev[n + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&R[n + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E[n+1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E_prev[n+1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&R[n+1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
#endif
    }


    // Fills in the LEFT Ghost Cells
    if (ranky == 0) {
      for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
      }
    } else {
      // communicate
#ifndef DISABLE_COMMUNICATION
      MPI_Isend(&E[1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&E_prev[1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&R[1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E[0 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E_prev[0 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&R[0 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
#endif
    }


    // Fills in the BOTTOM Ghost Cells
    if (rankx + 1 == cb.px) {
      for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
      }
    } else {
      // communicate
#ifndef DISABLE_COMMUNICATION
      MPI_Isend(&E[1 + (n+2)*m], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&E_prev[1 + (n+2)*m], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Isend(&R[1 + (n+2)*m], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E[1 + (n+2)*(m+1)], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_E, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&E_prev[1 + (n+2)*(m+1)], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      MPI_Irecv(&R[1 + (n+2)*(m+1)], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
#endif
    }

#ifndef DISABLE_COMMUNICATION
    vector<MPI_Status> status(reqs.size());
    MPI_Waitall(reqs.size(), reqs.data(), status.data());
#endif
    MPI_Barrier(MPI_COMM_WORLD);


//////////////////////////////////////////////////////////////////////////////

#define FUSED 1

#ifdef FUSED
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
	E_prev_tmp = E_prev + j;
        R_tmp = R + j;
	for(i = 0; i < n; i++) {
	    E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#else
    // Solve for the excitation, a PDE
    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
            E_prev_tmp = E_prev + j;
            for(i = 0; i < n; i++) {
                E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            }
    }

    /*
     * Solve the ODE, advancing excitation and recovery variables
     *     to the next timtestep
     */

    for(j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        E_tmp = E + j;
        R_tmp = R + j;
	E_prev_tmp = E_prev + j;
        for(i = 0; i < n; i++) {
	  E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	  R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif
     /////////////////////////////////////////////////////////////////////////////////

   if (cb.stats_freq){
     if ( !(niter % cb.stats_freq)){
        stats(E,m,n,&mx,&sumSq);
        double l2norm = L2Norm(sumSq);
        repNorms(l2norm,mx,dt,m,n,niter, cb.stats_freq);
    }
   }

   if (cb.plot_freq){
          if (!(niter % cb.plot_freq)){
	    plotter->updatePlot(E,  niter, m, n);
        }
    }

   // Swap current and previous meshes
   double *tmp = E; E = E_prev; E_prev = tmp;

 } //end of 'niter' loop at the beginning

  // gather information back
  stats(E_prev,m,n,&Linf,&sumSq);
  gather(m, n, sumSq, plotter, L2, Linf);

  // Swap pointers so we can re-use the arrays
  *_E = E;
  *_E_prev = E_prev;
}

void printMat2(const char mesg[], double *E, int m, int n){
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
