//	JEREMY BONNELL
//	CS 4230-001
//	PROGRAMMING ASSIGNMENT #5 - CUDA - SPARSE MATRIX/VECTOR MULTIPLY (CRS)
//
//	SEQUENTIAL AND CUDA VERSIONS
//
//	sparse_matvec.cu

#include <stdio.h>
#include <cutil.h>

#define __suif_min(x,y) ((x)<(y)?(x):(y))

;

#define N 1024
extern void MV_CRS_GPU_wrapper(float *, float *, float *, int *, int * , int, int, int);
extern int cudaMalloc();
extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
extern __global__ void mv_CRS_GPU(float *, float *, float *, int *, int *, int, int, int);
int blksz = 0;
int grdsz = 0;
int maxElement = 0;

void normalSparseMV(float *t, float *data, float *b, int *ptr, int *indices, int nr){
	//ptr stores which element in data entry is the first entry in a new row
	//indices says which column each data entry is in
	FILE *matrixCPU;
    matrixCPU = fopen("matrixCPU", "w");
	for (int i=0; i<nr; i++) {
		for (int j = ptr[i]; j<ptr[i+1]; j++) {
			t[i] = t[i] + data[j] * b[indices[j]];
		}
		fprintf(matrixCPU, "%f\n", t[i]); //stream results into file
	}
	fclose(matrixCPU);
}

extern void MV_CRS_GPU_wrapper(float *t, float *data, float *b, int *ptr, int *indices, int nr, int nc, int n)
{
    float *devO1Ptr;
    float *devI1Ptr;
    float *devI2Ptr;
	int *pPtr;
	int *iPtr;	
	
    cudaMalloc((void **)&devO1Ptr, nr * 4);
    cudaMemcpy(devO1Ptr, t, nr * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI1Ptr, n * 4);
    cudaMemcpy(devI1Ptr, data, n * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI2Ptr, nc * 4);
    cudaMemcpy(devI2Ptr, b, nc * 4, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&pPtr, (nr+1) * 4);
    cudaMemcpy(pPtr, ptr, (nr+1) * 4, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&iPtr, n * 4);
    cudaMemcpy(iPtr, indices, n * 4, cudaMemcpyHostToDevice);
    dim3 dimGrid(grdsz, 1);
    dim3 dimBlock(blksz, 1);
    mv_CRS_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, devI1Ptr, devI2Ptr, pPtr, iPtr, nr, maxElement, blksz);
    cudaMemcpy(t, devO1Ptr, nr * 4, cudaMemcpyDeviceToHost);
    cudaFree(devO1Ptr);
    cudaFree(devI1Ptr);
    cudaFree(devI2Ptr);
		
    return;
}

extern __global__ void mv_CRS_GPU(float *a, float *c, float *b, int *ptr, int *indices, int nr, int maxE, int blksz)
{
	
    int bx;
    int tx;
    float suif_tmp0;
    int j;
    int index;
	int endIndex;
	    	
    bx = blockIdx.x;
    tx = threadIdx.x;
    if (tx <= -(blksz * bx) + (nr-1))
      {
        suif_tmp0 = a[tx + blksz * bx];
        index = ptr[blksz * bx + tx];
		endIndex = ptr[blksz * bx + tx+1];
      }
	
		for (j = 0; j<maxE; j++){ 
			
			//ALWAYS HAVE TO SURROUND WITH THIS IF STMT TO KEEP UNWANTED THREADS OUT!
			if (tx <= -(blksz * bx) + (nr-1)){
				
				if((index + j) < endIndex)
					suif_tmp0 = suif_tmp0 + c[index+j] * b[indices[index+j]];
			}			
		}
		__syncthreads();
    
    if (tx <= -(blksz * bx) + (nr-1))
      {
        a[tx + blksz * bx] = suif_tmp0;
      }	  
}

main (int argc, char **argv) {
  FILE *fp;
  FILE *matrixGPU;
  char line[1024]; 
  int *ptr, *indices;
  float *t_h, *t_d, *b, *data;
  int n; // number of nonzero elements in data
  int nr; // number of rows in matrix
  int nc; // number of columns in matrix  
  int tempa=0;

  // create  CUDA event handles for timing purposes
  cudaEvent_t start_event, stop_event;
  float elapsed_time_seq, elapsed_time_gpu;
  float temp;

  // Open input file and read to end of comments
  if (argc !=2) abort(); 

  if ((fp = fopen(argv[1], "r")) == NULL) {
    abort();
  }

  fgets(line, 128, fp); //reads at most 128 chars per line
  while (line[0] == '%') {
    fgets(line, 128, fp); 
  }

  // Read number of rows (nr), number of columns (nc) and
  // number of elements and allocate memory for ptr, indices, data, b and t.
  sscanf(line,"%d %d %d\n", &nr, &nc, &n);
  ptr = (int *) malloc ((nr+1)*sizeof(int));
  indices = (int *) malloc(n*sizeof(int)); 
  t_h = (float *) malloc(nr*sizeof(float));   // h_a
  t_d = (float *) malloc(nr*sizeof(float));   // d_a
  b = (float *) malloc(nc*sizeof(float));     // b_cu
  data = (float *) malloc(n*sizeof(float));   // c
 
  // Calculate global vars blksz and gridsz  
  blksz = sqrt(nr) +(sqrt(nr)/2);
  temp = (float)nr/(float)blksz;
  grdsz = nr/blksz;
  if((temp - grdsz) > 0.0)
	  grdsz = (int)temp + 1;
  else
	  grdsz = (int)temp;
 
  // Read data in coordinate format and initialize sparse matrix
  int lastr=0;
  for (int i=0; i<n; i++) {
    int r;
    fscanf(fp,"%d %d %f\n", &r, &(indices[i]), &(data[i]));  
    indices[i]--;  // start numbering at 0
    if (r!=lastr) { 
      ptr[r-1] = i; 
      lastr = r; 
    }
  }
  ptr[nr] = n;
  
   // Calculate global max number of nonzero elements in all rows
  for (int i=0; i<nr; i++)
  {
	  tempa = ptr[i+1] - ptr[i];
	  maxElement = max(maxElement, tempa);
  } 

  // initialize t to 0 and b with random data  
  for (int i=0; i<nr; i++) {
    t_h[i] = 0.0;
    t_d[i] = 0.0;
  }

  for (int i=0; i<nc; i++) {
    b[i] = (float) rand()/1111111111;
  }

  fclose(fp);

  //------------------------------------SEQUENTIAL PORTION---------------------------------//

  // *******MAIN COMPUTATION, SEQUENTIAL VERSION***********//
  CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
  cudaEventRecord(start_event, 0);
  normalSparseMV(t_h, data, b, ptr, indices, nr);
  //normalMV(h_a, c, b_cu){
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time_seq,start_event, stop_event) )

  // TODO: Compute result on GPU and compare output

  //------------------------------------CUDA PORTION---------------------------------------//
  
  // *******MAIN COMPUTATION, CUDA VERSION******//
  CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
  cudaEventRecord(start_event, 0);   
  MV_CRS_GPU_wrapper(t_d, data, b, ptr, indices, nr, nc, n);
  //MV_GPU_wrapper(d_a, c, b_cu);  
  cudaThreadSynchronize();
  cudaEventRecord(stop_event, 0);
  cudaEventSynchronize(stop_event);
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time_gpu,start_event, stop_event) )

	  
  matrixGPU = fopen("matrixGPU", "w");
  for (int i=0; i<nr; i++) {
	  fprintf(matrixGPU, "%f\n", t_d[i]); //stream results into file
  }
  fclose(matrixGPU);
   

  CUTBoolean res = cutComparefe( t_h, t_d, nr, 0.01);
  if (res == 1) {
    printf("VALID!\n  Sequential Time: %.2f msec\n  Parallel Time: %.2f msec\n Speedup = %.2f\n", elapsed_time_seq, elapsed_time_gpu, elapsed_time_seq/elapsed_time_gpu);
  }
  else printf("INVALID...\n");
  
}
