#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include<string.h> 
#include<omp.h>
#include <time.h>

/* ************************************************************************** */
int Nd, Nc, Np;
double TOL; 
int thread_count; 

#define BUFSIZE 512

#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

// Define maximum number of iterations
#define MAX_ITER 1000


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  // Error Check
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0; double result; 
  char buffer[BUFSIZE], fileTag[BUFSIZE]; 
  
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      sk++;
      return result;
    }
  }

  if(sk==0){
    printf("ERROR! Could not find the tag: [%s] in the file [%s]\n", tag, fileName);
    exit(EXIT_FAILURE); 
  }
}


/* ************************************************************************** */
void readDataFile(char *fileName, double *data){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
  }

  int sk = 0;  
  char buffer[BUFSIZE], fileTag[BUFSIZE]; 
  
  int shift = Nd; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    if(Nd==2)
      sscanf(buffer, "%lf %lf", &data[sk*shift + 0], &data[sk*shift+1]);
    if(Nd==3)
      sscanf(buffer, "%lf %lf %lf", &data[sk*shift + 0], &data[sk*shift+1], &data[sk*shift+2]);
    if(Nd==4)
      sscanf(buffer, "%lf %lf %lf %lf", &data[sk*shift+0],&data[sk*shift+1], &data[sk*shift+2], &data[sk*shift+3]);
    sk++; 
  }
}

/* ************************************************************************** */
void writeDataFile(char *fileName, double *data, int *Ci){
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL) {
    printf("Error opening the output file\n");
  }

  for(int p=0; p<Np; p++){
    fprintf(fp, "%d %d ", p, Ci[p]);
    for(int dim=0; dim<Nd; dim++){
      fprintf(fp, "%.4f ", data[p*Nd + dim]);
    }
    fprintf(fp, "\n"); 
  }
  fclose(fp); 
}

/* ************************************************************************** */
void writeCentroidFile(char *fileName, double *Cm){
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL) {
    printf("Error opening the output file\n");
  }

  for(int n=0; n<Nc; n++){
    for(int dim=0; dim<Nd; dim++){
      fprintf(fp, "%.4f ", Cm[n*Nd + dim]);
    }
    fprintf(fp, "\n"); 
  }
  fclose(fp); 
}

/*************************************************************************** */
// Function to calculate Euclidean distance between two points
double distance(double *a, double *b) {
  double sum = 0.0; 
  for(int dim=0; dim < Nd; dim++){
    sum += pow((a[dim] - b[dim]), 2);
  }
  return sqrt(sum);
}


/*************************************************************************** */
// Function to assign each point to the nearest centroid
void assignPoints(double *data, int *Ci, int *Ck, double *Cm) {

  // Reset the number of points in the cluster
#pragma omp parallel for num_threads(thread_count)
  for(int n = 0; n < Nc; n++) {
    Ck[n] = 0;
  } 


  // Go over each point
#pragma omp parallel for num_threads(thread_count) reduction(+:Ck[:Nc])
  for (int p = 0; p < Np; p++) {
    double min_distance = INFINITY;
    int cluster_index   = 0;

    for (int n = 0; n < Nc; n++){
      double d = distance(&data[p*Nd], &Cm[n*Nd]);
      if (d < min_distance) {
        min_distance  = d;
        cluster_index = n;
      }
    }
    Ck[cluster_index] +=1;
    Ci[p]              = cluster_index;  
  }
}

/*************************************************************************** */
// Function to update centroids based on the mean of assigned points
double updateCentroids(double *data, int *Ci, int *Ck, double *Cm) {

double *CmCopy = (double *)malloc(Nc*Nd*sizeof(double));

#pragma omp parallel for num_threads(thread_count)	\
  default(none) shared(CmCopy, Cm, Nc, Nd) collapse(2)
  for(int n = 0; n < Nc; n++) {
    for(int dim = 0; dim<Nd; dim++){
      CmCopy[n*Nd + dim] =  Cm[n*Nd + dim]; 
    }
  } 

#pragma omp parallel for num_threads(thread_count) collapse(2)
  for(int n = 0; n < Nc; n++) {
    for(int dim = 0; dim<Nd; dim++){
      Cm[n*Nd + dim] = 0.0; 
    }
  } 


#pragma omp parallel for num_threads(thread_count) reduction(+:Cm[:Nc*Nd])
  for(int p=0; p<Np; p++){
    // Get cluster of the point
    int cluster_index = Ci[p]; 
    double sum = 0.0; 
    for(int dim = 0; dim<Nd; dim++){
      // reduction required here! 
      Cm[cluster_index*Nd + dim] += data[p*Nd + dim];
    }
  } 

double err = 1.E-12; 
#pragma omp parallel for num_threads(thread_count) reduction(max:err)
  for (int n = 0; n < Nc; n++){
    for(int dim = 0; dim<Nd; dim++){
      Cm[n*Nd + dim] = Cm[n*Nd + dim]/Ck[n]; 
      err = MAX(err, fabs(Cm[n*Nd + dim] - CmCopy[n*Nd + dim])); 
    }
  }

  free(CmCopy);
  return err;   

}

/*************************************************************************** */
// Function to perform k-means clustering
void kMeans(double *data, int *Ci, int *Ck, double *Cm) {

  // Initialize clusters randomly
  for (int n = 0; n < Nc; n++) {
    int ids = rand() % Np;

    for(int dim = 0; dim<Nd; dim++){
      Cm[n*Nd + dim] = data[ids*Nd+ dim];  
    }
    Ck[n]   = 0; 
    Ci[ids] = n; 
  }

  double err = INFINITY; 

  int sk = 0; 

  double tstart = omp_get_wtime(); 
  while(err>TOL) {
    assignPoints(data, Ci, Ck, Cm);
    err = updateCentroids(data, Ci, Ck, Cm);
    printf("\r Iteration %d %.12e\n", sk, err);sk++; 
    fflush(stdout);
  }

  printf("\n");
 
}

/*************************************************************************** */

int main ( int argc, char *argv[] ){
  if(argc!=3){
    printf("Usage: ./kmeans input.dat data.dat\n");
    return -1;  
  }

  // Read Number of Data Points 
  Np = (int) readInputFile(argv[1], "NUMBER_OF_POINTS");
  // Read Number of clusters
  Nc = (int) readInputFile(argv[1], "NUMBER_OF_CLUSTERS");
  // Read Dimension of Data
  Nd = (int) readInputFile(argv[1], "DATA_DIMENSION");
  TOL = readInputFile(argv[1], "TOLERANCE");

  thread_count = (int) readInputFile(argv[1], "NUMBER_OF_THREADS");

  omp_set_num_threads(thread_count); 

  // Allocate data [c_i, x_i, y_i, z_i, ...]
  double *data = (double*) malloc(Np*Nd*sizeof(double));
  // Cluster id for every point
  int      *Ci = (int *) calloc(Np, sizeof(int));
  // Number of data points in every cluster
  int      *Ck = (int *) calloc(Nc, sizeof(int));
  // Centroid of every clusters
  double   *Cm = (double*) calloc(Nc*Nd, sizeof(double));

  // Centroid of every clusters

  // Fill point data from file
  readDataFile(argv[2], data);  

  double start = omp_get_wtime();
  // Perform k-means clustering
  kMeans(data, Ci, Ck, Cm);
  double elapsed_time = omp_get_wtime() - start; 
    

  int rank = omp_get_thread_num();
  if(rank==0)
    printf("Total Compute Time: %.8e using %d threads\n", elapsed_time, thread_count);

  // Report Results
  for(int n=0; n<Nc; n++){
    int Npoints =Ck[n]; 
    printf("(%d of %d) points are in the cluster %d with centroid( ", Npoints, Np, n);
    for(int dim = 0; dim<Nd; dim++){
      printf("%f ", Cm[n*Nd + dim]); 
    }
    printf(") \n"); 
  }

  writeDataFile("output.dat",    data, Ci);
  writeCentroidFile("centroids.dat", Cm);


  free(data);
  free(Ci);
  free(Ck);
  free(Cm);

  return 0;
}
