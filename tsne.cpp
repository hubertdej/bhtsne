/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include "tsne.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using namespace std;

static double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

static void zeroMean(double* X, int N, int D);
static void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity);
static double randn();
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC);
static double evaluateError(double* P, double* Y, int N, int D);
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD);

// Perform t-SNE
void TSNE::run(double* X, int N, int D, double* Y, int no_dims, double perplexity, double theta, int rand_seed,
               bool skip_random_init, int max_iter, int stop_lying_iter, int mom_switch_iter) {
  // Set random seed
  if (skip_random_init != true) {
    if (rand_seed >= 0) {
      printf("Using random seed: %d\n", rand_seed);
      srand((unsigned int)rand_seed);
    } else {
      printf("Using current time as random seed...\n");
      srand(time(NULL));
    }
  }

  // Determine whether we are using an exact algorithm
  if (N - 1 < 3 * perplexity) {
    printf("Perplexity too large for the number of data points!\n");
    exit(1);
  }
  printf("Using no_dims = %d, perplexity = %f, and theta = %f\n", no_dims, perplexity, theta);

  // Set learning parameters
  float total_time = .0;
  clock_t start, end;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;

  // Allocate some memory
  double* dY = (double*)malloc(N * no_dims * sizeof(double));
  double* uY = (double*)malloc(N * no_dims * sizeof(double));
  double* gains = (double*)malloc(N * no_dims * sizeof(double));
  if (dY == NULL || uY == NULL || gains == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  for (int i = 0; i < N * no_dims; i++) uY[i] = .0;
  for (int i = 0; i < N * no_dims; i++) gains[i] = 1.0;

  // Normalize input data (to prevent numerical problems)
  printf("Computing input similarities...\n");
  start = clock();
  zeroMean(X, N, D);
  double max_X = .0;
  for (int i = 0; i < N * D; i++) {
    if (fabs(X[i]) > max_X) max_X = fabs(X[i]);
  }
  for (int i = 0; i < N * D; i++) X[i] /= max_X;

  // Compute input similarities for exact t-SNE
  double* P;
  unsigned int* row_P;
  unsigned int* col_P;
  double* val_P;

  // Compute similarities
  P = (double*)malloc(N * N * sizeof(double));
  if (P == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  computeGaussianPerplexity(X, N, D, P, perplexity);

  // Symmetrize input similarities
  printf("Symmetrizing...\n");
  int nN = 0;
  for (int n = 0; n < N; n++) {
    int mN = (n + 1) * N;
    for (int m = n + 1; m < N; m++) {
      P[nN + m] += P[mN + n];
      P[mN + n] = P[nN + m];
      mN += N;
    }
    nN += N;
  }
  double sum_P = .0;
  for (int i = 0; i < N * N; i++) sum_P += P[i];
  for (int i = 0; i < N * N; i++) P[i] /= sum_P;
  end = clock();

  // Lie about the P-values
  for (int i = 0; i < N * N; i++) P[i] *= 12.0;

  // Initialize solution (randomly)
  if (skip_random_init != true) {
    for (int i = 0; i < N * no_dims; i++) Y[i] = randn() * .0001;
  }

  // Perform main training loop
  printf("Input similarities computed in %4.2f seconds!\nLearning embedding...\n", (float)(end - start) / CLOCKS_PER_SEC);
  start = clock();

  for (int iter = 0; iter < max_iter; iter++) {
    // Compute (approximate) gradient
    computeExactGradient(P, Y, N, no_dims, dY);

    // Update gains
    for (int i = 0; i < N * no_dims; i++) gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    for (int i = 0; i < N * no_dims; i++)
      if (gains[i] < .01) gains[i] = .01;

    // Perform gradient update (with momentum and gains)
    for (int i = 0; i < N * no_dims; i++) uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    for (int i = 0; i < N * no_dims; i++) Y[i] = Y[i] + uY[i];

    // Make solution zero-mean
    zeroMean(Y, N, no_dims);

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
      for (int i = 0; i < N * N; i++) P[i] /= 12.0;
    }
    if (iter == mom_switch_iter) momentum = final_momentum;

    // Print out progress
    if (iter > 0 && (iter % 50 == 0 || iter == max_iter - 1)) {
      end = clock();
      double C = .0;
      C = evaluateError(P, Y, N, no_dims);
      if (iter == 0)
        printf("Iteration %d: error is %f\n", iter + 1, C);
      else {
        total_time += (float)(end - start) / CLOCKS_PER_SEC;
        printf("Iteration %d: error is %f (50 iterations in %4.2f seconds)\n", iter, C, (float)(end - start) / CLOCKS_PER_SEC);
      }
      start = clock();
    }
  }
  end = clock();
  total_time += (float)(end - start) / CLOCKS_PER_SEC;

  // Clean up memory
  free(dY);
  free(uY);
  free(gains);
  free(P);
  printf("Fitting performed in %4.2f seconds.\n", total_time);
}

// Compute gradient of the t-SNE cost function (exact)
static void computeExactGradient(double* P, double* Y, int N, int D, double* dC) {
  // Make sure the current gradient contains zeros
  for (int i = 0; i < N * D; i++) dC[i] = 0.0;

  // Compute the squared Euclidean distance matrix
  double* DD = (double*)malloc(N * N * sizeof(double));
  if (DD == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  computeSquaredEuclideanDistance(Y, N, D, DD);

  // Compute Q-matrix and normalization sum
  double* Q = (double*)malloc(N * N * sizeof(double));
  if (Q == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  double sum_Q = .0;
  int nN = 0;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < N; m++) {
      if (n != m) {
        Q[nN + m] = 1 / (1 + DD[nN + m]);
        sum_Q += Q[nN + m];
      }
    }
    nN += N;
  }

  // Perform the computation of the gradient
  nN = 0;
  int nD = 0;
  for (int n = 0; n < N; n++) {
    int mD = 0;
    for (int m = 0; m < N; m++) {
      if (n != m) {
        double mult = (P[nN + m] - (Q[nN + m] / sum_Q)) * Q[nN + m];
        for (int d = 0; d < D; d++) {
          dC[nD + d] += (Y[nD + d] - Y[mD + d]) * mult;
        }
      }
      mD += D;
    }
    nN += N;
    nD += D;
  }

  // Free memory
  free(DD);
  DD = NULL;
  free(Q);
  Q = NULL;
}

// Evaluate t-SNE cost function (exactly)
static double evaluateError(double* P, double* Y, int N, int D) {
  // Compute the squared Euclidean distance matrix
  double* DD = (double*)malloc(N * N * sizeof(double));
  double* Q = (double*)malloc(N * N * sizeof(double));
  if (DD == NULL || Q == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  computeSquaredEuclideanDistance(Y, N, D, DD);

  // Compute Q-matrix and normalization sum
  int nN = 0;
  double sum_Q = DBL_MIN;
  for (int n = 0; n < N; n++) {
    for (int m = 0; m < N; m++) {
      if (n != m) {
        Q[nN + m] = 1 / (1 + DD[nN + m]);
        sum_Q += Q[nN + m];
      } else
        Q[nN + m] = DBL_MIN;
    }
    nN += N;
  }
  for (int i = 0; i < N * N; i++) Q[i] /= sum_Q;

  // Sum t-SNE error
  double C = .0;
  for (int n = 0; n < N * N; n++) {
    C += P[n] * log((P[n] + FLT_MIN) / (Q[n] + FLT_MIN));
  }

  // Clean up memory
  free(DD);
  free(Q);
  return C;
}

// Compute input similarities with a fixed perplexity
static void computeGaussianPerplexity(double* X, int N, int D, double* P, double perplexity) {
  // Compute the squared Euclidean distance matrix
  double* DD = (double*)malloc(N * N * sizeof(double));
  if (DD == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  computeSquaredEuclideanDistance(X, N, D, DD);

  // Compute the Gaussian kernel row by row
  int nN = 0;
  for (int n = 0; n < N; n++) {
    // Initialize some variables
    bool found = false;
    double beta = 1.0;
    double min_beta = -DBL_MAX;
    double max_beta = DBL_MAX;
    double tol = 1e-5;
    double sum_P;

    // Iterate until we found a good perplexity
    int iter = 0;
    while (!found && iter < 200) {
      // Compute Gaussian kernel row
      for (int m = 0; m < N; m++) P[nN + m] = exp(-beta * DD[nN + m]);
      P[nN + n] = DBL_MIN;

      // Compute entropy of current row
      sum_P = DBL_MIN;
      for (int m = 0; m < N; m++) sum_P += P[nN + m];
      double H = 0.0;
      for (int m = 0; m < N; m++) H += beta * (DD[nN + m] * P[nN + m]);
      H = (H / sum_P) + log(sum_P);

      // Evaluate whether the entropy is within the tolerance level
      double Hdiff = H - log(perplexity);
      if (Hdiff < tol && -Hdiff < tol) {
        found = true;
      } else {
        if (Hdiff > 0) {
          min_beta = beta;
          if (max_beta == DBL_MAX || max_beta == -DBL_MAX)
            beta *= 2.0;
          else
            beta = (beta + max_beta) / 2.0;
        } else {
          max_beta = beta;
          if (min_beta == -DBL_MAX || min_beta == DBL_MAX) {
            if (beta < 0) {
              beta *= 2;
            } else {
              beta = beta <= 1.0 ? -0.5 : beta / 2.0;
            }
          } else {
            beta = (beta + min_beta) / 2.0;
          }
        }
      }

      // Update iteration counter
      iter++;
    }

    // Row normalize P
    for (int m = 0; m < N; m++) P[nN + m] /= sum_P;
    nN += N;
  }

  // Clean up memory
  free(DD);
  DD = NULL;
}

// Compute squared Euclidean distance matrix
static void computeSquaredEuclideanDistance(double* X, int N, int D, double* DD) {
  const double* XnD = X;
  for (int n = 0; n < N; ++n, XnD += D) {
    const double* XmD = XnD + D;
    double* curr_elem = &DD[n * N + n];
    *curr_elem = 0.0;
    double* curr_elem_sym = curr_elem + N;
    for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N) {
      *(++curr_elem) = 0.0;
      for (int d = 0; d < D; ++d) {
        *curr_elem += (XnD[d] - XmD[d]) * (XnD[d] - XmD[d]);
      }
      *curr_elem_sym = *curr_elem;
    }
  }
}

// Makes data zero-mean
static void zeroMean(double* X, int N, int D) {
  // Compute data mean
  double* mean = (double*)calloc(D, sizeof(double));
  if (mean == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  int nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      mean[d] += X[nD + d];
    }
    nD += D;
  }
  for (int d = 0; d < D; d++) {
    mean[d] /= (double)N;
  }

  // Subtract data mean
  nD = 0;
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < D; d++) {
      X[nD + d] -= mean[d];
    }
    nD += D;
  }
  free(mean);
  mean = NULL;
}

// Generates a Gaussian random number
static double randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while ((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool TSNE::load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {
  // Open file, read first 2 integers, allocate memory, and read the data
  FILE* h;
  if ((h = fopen("data.dat", "r+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return false;
  }
  fread(n, sizeof(int), 1, h);              // number of datapoints
  fread(d, sizeof(int), 1, h);              // original dimensionality
  fread(theta, sizeof(double), 1, h);       // gradient accuracy
  fread(perplexity, sizeof(double), 1, h);  // perplexity
  fread(no_dims, sizeof(int), 1, h);        // output dimensionality
  fread(max_iter, sizeof(int), 1, h);       // maximum number of iterations
  *data = (double*)malloc(*d * *n * sizeof(double));
  if (*data == NULL) {
    printf("Memory allocation failed!\n");
    exit(1);
  }
  fread(*data, sizeof(double), *n * *d, h);           // the data
  if (!feof(h)) fread(rand_seed, sizeof(int), 1, h);  // random seed
  fclose(h);
  printf("Read the %i x %i data matrix successfully!\n", *n, *d);
  return true;
}

// Function that saves map to a t-SNE file
void TSNE::save_data(double* data, int* landmarks, double* costs, int n, int d) {
  // Open file, write first 2 integers and then the data
  FILE* h;
  if ((h = fopen("result.dat", "w+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return;
  }
  fwrite(&n, sizeof(int), 1, h);
  fwrite(&d, sizeof(int), 1, h);
  fwrite(data, sizeof(double), n * d, h);
  fwrite(landmarks, sizeof(int), n, h);
  fwrite(costs, sizeof(double), n, h);
  fclose(h);
  printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
