#include <cstdio>
#include <cstdlib>

#include "tsne.h"

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
