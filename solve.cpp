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
#include <string.h>
#include <malloc.h>
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

static double** mem_E_tmp;
static double** mem_E_prev_tmp;
static double** mem_R_tmp;

#define ALIGNMENT 256
#ifndef BLOCKING
  #define BLOCKING 1
#endif
#ifndef FUSED
  #define FUSED 1
#endif
#ifndef BLOCK_SIZE_0
#define BLOCK_SIZE_0 500
#endif
#ifndef BLOCK_SIZE_1
#define BLOCK_SIZE_1 50
#endif

#if BLOCKING
static inline void copy_blk_pad (double* dst, int dh, int dw, int dlda, double* src, int sh, int sw, int slda) {
  // if (dw>sw || dh>sh) memset(dst, 0, dh*dw*sizeof(double));
  for (int i=0; i<sh; i++) {
    memcpy(dst+i*dlda, src+i*slda, sw*sizeof(double));
    if (dw>sw) memset(dst+i*dlda+sw, 0, (dw-sw)*sizeof(double));
  }
  if (dh > sh) memset(dst+sh*dlda, 0, (dh-sh)*dw*sizeof(double));
}

// do_block_0_fused(blk_E_prev_tmp, blk_R_tmp, blk_E_tmp, BLOCK_SIZE_0, BLOCK_SIZE_0, BLOCK_SIZE_0);
static inline void do_block_1_fused(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
  const int innerBlockRowStartIndex = (n+2)+1;
  const int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  for(int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
      double *E_tmp = E + j;
      double *E_prev_tmp = E_prev + j;
      double *R_tmp = R + j;
      for(int i = 0; i < n; i++) {
          E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
          E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
          R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
      }
  }
}

static inline void do_block_0_fused(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
    double * blk_E_prev_tmp = mem_E_prev_tmp[0];
    double * blk_R_tmp = mem_R_tmp[0];
    double * blk_E_tmp = mem_E_tmp[0];
    for (int i=1; i<=m; i+=BLOCK_SIZE_1) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_1) {
        const int M = min (BLOCK_SIZE_1+2, m+3-i);
        const int N = min (BLOCK_SIZE_1+2, n+3-j);

        copy_blk_pad(blk_E_prev_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, E_prev + (i-1)*lda + j-1, M, N, lda);
        copy_blk_pad(blk_R_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, R + (i-1)*lda + j-1, M, N, lda);

        do_block_1_fused(blk_E_prev_tmp, blk_R_tmp, blk_E_tmp, BLOCK_SIZE_1, BLOCK_SIZE_1, BLOCK_SIZE_1+2, alpha, dt);

        copy_blk_pad(E+i*lda+j, M-2, N-2, lda, blk_E_tmp+(BLOCK_SIZE_1+2)+1, M-2, N-2, BLOCK_SIZE_1+2);
        copy_blk_pad(R+i*lda+j, M-2, N-2, lda, blk_R_tmp+(BLOCK_SIZE_1+2)+1, M-2, N-2, BLOCK_SIZE_1+2);
      }
    }
}

static inline void do_block_1_unfused_A(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
  // assert(lda == n+2);
  const int innerBlockRowStartIndex = (n+2)+1;
  const int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  for(int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
      double *E_tmp = E + j;
      double *E_prev_tmp = E_prev + j;
      double *R_tmp = R + j;
      for(int i = 0; i < n; i++) {
        E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
      }
  }
}

static inline void do_block_1_unfused_B(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
  // assert(lda == n+2);
  const int innerBlockRowStartIndex = (n+2)+1;
  const int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  for(int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
      double *E_tmp = E + j;
      double *E_prev_tmp = E_prev + j;
      double *R_tmp = R + j;
      for(int i = 0; i < n; i++) {
        E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
	      R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
      }
  }
}



static inline void do_block_0_unfused_A(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
    double * blk_E_prev_tmp = mem_E_prev_tmp[0];
    double * blk_R_tmp = mem_R_tmp[0];
    double * blk_E_tmp = mem_E_tmp[0];
    for (int i=1; i<=m; i+=BLOCK_SIZE_1) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_1) {
        const int M = min (BLOCK_SIZE_1+2, m+3-i);
        const int N = min (BLOCK_SIZE_1+2, n+3-j);

        copy_blk_pad(blk_E_prev_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, E_prev + (i-1)*lda + j-1, M, N, lda);

        do_block_1_unfused_A(blk_E_prev_tmp, blk_R_tmp, blk_E_tmp, BLOCK_SIZE_1, BLOCK_SIZE_1, BLOCK_SIZE_1+2, alpha, dt);

        copy_blk_pad(E+i*lda+j, M-2, N-2, lda, blk_E_tmp+(BLOCK_SIZE_1+2)+1, M-2, N-2, BLOCK_SIZE_1+2);
      }
    }
}

static inline void do_block_0_unfused_B(double *E_prev, double *R, double *E,
                                    int m, int n, int lda, double alpha, double dt) {
    double * blk_E_prev_tmp = mem_E_prev_tmp[0];
    double * blk_R_tmp = mem_R_tmp[0];
    double * blk_E_tmp = mem_E_tmp[0];
    for (int i=1; i<=m; i+=BLOCK_SIZE_1) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_1) {
        const int M = min (BLOCK_SIZE_1+2, m+3-i);
        const int N = min (BLOCK_SIZE_1+2, n+3-j);

        copy_blk_pad(blk_E_prev_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, E_prev + (i-1)*lda + j-1, M, N, lda);
        copy_blk_pad(blk_E_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, E + (i-1)*lda + j-1, M, N, lda);
        copy_blk_pad(blk_R_tmp, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, BLOCK_SIZE_1+2, R + (i-1)*lda + j-1, M, N, lda);

        do_block_1_unfused_B(blk_E_prev_tmp, blk_R_tmp, blk_E_tmp, BLOCK_SIZE_1, BLOCK_SIZE_1, BLOCK_SIZE_1+2, alpha, dt);

        copy_blk_pad(E+i*lda+j, M-2, N-2, lda, blk_E_tmp+(BLOCK_SIZE_1+2)+1, M-2, N-2, BLOCK_SIZE_1+2);
        copy_blk_pad(R+i*lda+j, M-2, N-2, lda, blk_R_tmp+(BLOCK_SIZE_1+2)+1, M-2, N-2, BLOCK_SIZE_1+2);
      }
    }
}
#endif /* BLOCKING */

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
  const int innerBlockRowStartIndex = (n+2)+1;
  const int innerBlockRowEndIndex = (((m+2)*(n+2) - 1) - (n)) - (n+2);

  MPI_Request req;

  MPI_Datatype bufferTypeRow;
  MPI_Datatype bufferTypeColumn;
  MPI_Type_vector(n, 1, 1, MPI_DOUBLE, &bufferTypeRow); // row buffer
  MPI_Type_vector(m, 1, n + 2, MPI_DOUBLE, &bufferTypeColumn); // column buffer
  MPI_Type_commit(&bufferTypeRow);
  MPI_Type_commit(&bufferTypeColumn);

#if BLOCKING
  mem_E_tmp = (double **)malloc(4*sizeof(double*));
  mem_E_prev_tmp = (double **)malloc(4*sizeof(double*));
  mem_R_tmp = (double **)malloc(4*sizeof(double*));

  if (posix_memalign((void **) &mem_E_tmp[0], ALIGNMENT, (BLOCK_SIZE_1+2)*(BLOCK_SIZE_1+2)*sizeof(double)) != 0) {
      printf("[ERROR] posix_memalign: failed in REDUCE_CONFLICT_3 (&mem_E_tmp)\n");
  }
  if (posix_memalign((void **) &mem_E_prev_tmp[0], ALIGNMENT, (BLOCK_SIZE_1+2)*(BLOCK_SIZE_1+2)*sizeof(double)) != 0) {
      printf("[ERROR] posix_memalign: failed in REDUCE_CONFLICT_3 (&mem_E_prev_tmp)\n");
  }
  if (posix_memalign((void **) &mem_R_tmp[0], ALIGNMENT, (BLOCK_SIZE_1+2)*(BLOCK_SIZE_1+2)*sizeof(double)) != 0) {
      printf("[ERROR] posix_memalign: failed in REDUCE_CONFLICT_3 (&mem_R_tmp)\n");
  }
#endif

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
    if (cb.noComm || rankx == 0) {
      for (i = 0; i < (n+2); i++) {
        E_prev[i] = E_prev[i + (n+2)*2];
      }
    } else if (!cb.noComm) {
      // communicate
      MPI_Isend(&E_prev[1 + (n+2)*1], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Isend(&R[1 + (n+2)*1], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
      MPI_Irecv(&E_prev[1 + (n+2)*0], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Irecv(&R[1 + (n+2)*0], 1, bufferTypeRow, composeRank(rankx - 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
    }

    // Fills in the RIGHT Ghost Cells
    if (cb.noComm || ranky + 1 == cb.py) {
      for (i = (n+1); i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i-2];
      }
    } else if (!cb.noComm) {
      // communicate
      MPI_Isend(&E_prev[n + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Isend(&R[n + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
      MPI_Irecv(&E_prev[n+1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Irecv(&R[n+1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky + 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
    }


    // Fills in the LEFT Ghost Cells
    if (cb.noComm || ranky == 0) {
      for (i = 0; i < (m+2)*(n+2); i+=(n+2)) {
        E_prev[i] = E_prev[i+2];
      }
    } else if (!cb.noComm) {
      // communicate
      MPI_Isend(&E_prev[1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Isend(&R[1 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
      MPI_Irecv(&E_prev[0 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Irecv(&R[0 + (n+2)*1], 1, bufferTypeColumn, composeRank(rankx, ranky - 1, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
    }


    // Fills in the BOTTOM Ghost Cells
    if (cb.noComm || rankx + 1 == cb.px) {
      for (i = ((m+2)*(n+2)-(n+2)); i < (m+2)*(n+2); i++) {
        E_prev[i] = E_prev[i - (n+2)*2];
      }
    } else if (!cb.noComm) {
      // communicate
      MPI_Isend(&E_prev[1 + (n+2)*m], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Isend(&R[1 + (n+2)*m], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
      MPI_Irecv(&E_prev[1 + (n+2)*(m+1)], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_EP, MPI_COMM_WORLD, &req);
      reqs.push_back(req);
      // MPI_Irecv(&R[1 + (n+2)*(m+1)], 1, bufferTypeRow, composeRank(rankx + 1, ranky, cb.px, cb.py), TAG_COMP_R, MPI_COMM_WORLD, &req);
      // reqs.push_back(req);
    }

    if (!cb.noComm) {
      vector<MPI_Status> status(reqs.size());
      MPI_Waitall(reqs.size(), reqs.data(), status.data());
    }
    MPI_Barrier(MPI_COMM_WORLD);


//////////////////////////////////////////////////////////////////////////////

#if FUSED
    // Solve for the excitation, a PDE


#if BLOCKING
    for (int i=1; i<=m; i+=BLOCK_SIZE_0) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_0) {
        const int offset = (i-1)*(n+2)+(j-1);
        const int M = min (BLOCK_SIZE_0, m+1-i);
        const int N = min (BLOCK_SIZE_0, n+1-j);
        do_block_0_fused(E_prev+offset, R+offset, E+offset, M, N, n+2, alpha, dt);
      }
    }
#else // BLOCKING
    for(int j = innerBlockRowStartIndex; j <= innerBlockRowEndIndex; j+=(n+2)) {
        double *E_tmp = E + j;
        double *E_prev_tmp = E_prev + j;
        double *R_tmp = R + j;
        for(int i = 0; i < n; i++) {
            E_tmp[i] = E_prev_tmp[i]+alpha*(E_prev_tmp[i+1]+E_prev_tmp[i-1]-4*E_prev_tmp[i]+E_prev_tmp[i+(n+2)]+E_prev_tmp[i-(n+2)]);
            E_tmp[i] += -dt*(kk*E_prev_tmp[i]*(E_prev_tmp[i]-a)*(E_prev_tmp[i]-1)+E_prev_tmp[i]*R_tmp[i]);
            R_tmp[i] += dt*(epsilon+M1* R_tmp[i]/( E_prev_tmp[i]+M2))*(-R_tmp[i]-kk*E_prev_tmp[i]*(E_prev_tmp[i]-b-1));
        }
    }
#endif // BLOCKING
#else // FUSED
#if BLOCKING
    for (int i=1; i<=m; i+=BLOCK_SIZE_0) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_0) {
        const int offset = (i-1)*(n+2)+(j-1);
        const int M = min (BLOCK_SIZE_0, m+1-i);
        const int N = min (BLOCK_SIZE_0, n+1-j);
        do_block_0_unfused_A(E_prev+offset, R+offset, E+offset, M, N, n+2, alpha, dt);
      }
    }
    for (int i=1; i<=m; i+=BLOCK_SIZE_0) {
      for (int j=1; j<=n; j+=BLOCK_SIZE_0) {
        const int offset = (i-1)*(n+2)+(j-1);
        const int M = min (BLOCK_SIZE_0, m+1-i);
        const int N = min (BLOCK_SIZE_0, n+1-j);
        do_block_0_unfused_B(E_prev+offset, R+offset, E+offset, M, N, n+2, alpha, dt);
      }
    }
#else // BLOCKING
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
#endif // BLOCKING
#endif // FUSED
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
  // gather(m, n, sumSq, plotter, L2, Linf);

  double tmpLinf = Linf, tmpSumSq = sumSq;
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Reduce(&tmpLinf, &Linf, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&tmpSumSq, &sumSq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  L2 = L2Norm(sumSq);

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
