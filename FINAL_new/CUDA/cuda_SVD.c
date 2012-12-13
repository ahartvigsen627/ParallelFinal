#include <stdio.h>
#include <cutil.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>

#define epsilon 1.e-8

using namespace std;

//may be able to run stuff faster, if transpose some things b4 reduce and rotate. i.e. Betas
//also. see how he initializes things in separate loops. probably faster because don't have to keep
//loading different arrays into cache. work on one completely, then move on to next.

#define __suif_min(x,y) ((x)<(y)?(x):(y))
#define __suif_max(x,y) ((x)>(y)?(x):(y))
//#define SIZE 512

extern void ROTATE_GPU_wrapper(double *, double *, double *, int, int);
extern int cudaMalloc();
extern int cudaMemcpy();
extern int cudaFree();
extern void __syncthreads();
extern int cudaMemcpyToSymbol();
extern __global__ void rotate_GPU(double *, double *, double *, int, int, int, int);
int blksz = 0;
int grdsz = 0;
int maxElement = 0;

template <typename T> double sgn(T val)//double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

extern void ROTATE_GPU_wrapper(double *u, double *b, double *g, int n, int i)
{
    double *devO1Ptr;
    double *devI1Ptr;
    double *devI2Ptr;
				
    cudaMalloc((void **)&devO1Ptr, n*n*8);
    cudaMemcpy(devO1Ptr, u, n*n*8, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI1Ptr, n * 8);
    cudaMemcpy(devI1Ptr, b, n * 8, cudaMemcpyHostToDevice);
    cudaMalloc((void **)&devI2Ptr, n * 8);
    cudaMemcpy(devI2Ptr, g, n * 8, cudaMemcpyHostToDevice);
	dim3 dimGrid(grdsz, 1);
    dim3 dimBlock(blksz, 1);
    rotate_GPU<<<dimGrid,dimBlock>>>(devO1Ptr, devI1Ptr, devI2Ptr, n, i, blksz, grdsz);
    cudaMemcpy(b, devI1Ptr, n * 8, cudaMemcpyDeviceToHost);
	cudaMemcpy(g, devI2Ptr, n * 8, cudaMemcpyDeviceToHost);
	cudaFree(devO1Ptr);
    cudaFree(devI1Ptr);
    cudaFree(devI2Ptr);
	return;
}

extern __global__ void rotate_GPU(double *u, double *b, double *g, int n, int i, int blksz, int grdsz)
{
    //float suif_tmp0;
    __shared__ double _P1[96];//192 big enough for 16k x 16k matrix. 96 is for 4096
  
	int bx = blockIdx.x;
    int tx = threadIdx.x;
	double gamma, beta;
	int rowIndex;
	
    if (tx <= -(blksz * bx) + (n-1))
      {		  
		  rowIndex = blksz*bx + tx;
		  beta  = 0.0;
		  gamma = 0.0;  
      }
			
		for (int k = 0; k < grdsz; k++)
		{
			if(tx <= -(blksz * k) + (n-1)) 
			  {
				//_P1[tx] = ((double (*)[SIZE])u)[i][blksz * k + tx];
				  _P1[tx] = u[blksz * k + tx + (i)*n]; //gamma is MV of the ith row of U_t
			  }
		  
			__syncthreads();
			if(tx <= -(blksz * bx) + (n-1)) // (rowIndex < i)
			  {
				if(rowIndex <= i)
				for (int j = blksz * k; j <=__suif_min(blksz*k + blksz-1, (n-1)); j++)
				{
					//THIS IS AN EXAMPLE OF HOW A CONTIGUOUSLY STORED MATRIX CAN BE INDEXED!!! CAST THEN 2D indexing
					//OR 1D INDEXING BY *n ON ROW and +j
					//beta += (((double (*)[SIZE])u)[rowIndex][j])*(((double (*)[SIZE])u)[rowIndex][j]);
					//gamma += (((double (*)[SIZE])u)[rowIndex][j]) * (_P1[j - blksz * k]);
					beta += (u[rowIndex*n+j])*(u[rowIndex*n+j]);
					gamma += (u[rowIndex*n+j]) * (_P1[j - blksz * k]);
				}
			  }     
			__syncthreads();
		}		

		  if (tx <= -(blksz * bx) + (n-1))
		  {	
			  if(rowIndex <= i)
			  {	
				  b[rowIndex] = beta;
				  g[rowIndex] = gamma;
			  }			  
		  }
}


// start main() ---------------------------------------------------------------------------------
main (int argc, char **argv) {

  // SVD setup initialization
  int M,N;
  string T,P,Db;
  M = atoi(argv[1]);
  N = atoi(argv[2]);
  double elapsedTime;//,elapsedTime2;
  timeval start,end;//,end2;
  double **V_t, **V, **A, **U, **Uu, **G;
  double *U_t, *S;

  if(argc < 4){
	  cout<<"Please input the size of Matrix and at least one of the options: -t -p -d";
	  return 0;
  }
   
  if(M != N){
	  cout<<"Error: Matrix must be square";
	  return 0;
  }
  
  if(argc > 3){
    T = argv[3];
    if(argc > 4){
      P = argv[4];
      if(argc > 5){
        Db = argv[5];
      }
    }
  }

  //U_t = new double*[N];
  U_t = new double[N*N];
  U = new double*[N];
  V = new double*[N];
  S = new double[N];
  //uiVec = new double[N];
  V_t = new double*[N];
  A = new double*[N];
  Uu = new double*[N];
  G = new double*[N];
  
  for(int i =0; i<N; i++){
	//U_t[i] = new float[N];
	U[i] = new double[N];
 	V[i] = new double[N];	
	V_t[i] = new double[N];
	A[i] = new double[N];
	Uu[i] = new double[N];
	G[i] = new double[N];
  } 
  
  //Read from file matrix, if not available, app quit
  //Already transposed
  ifstream matrixfile("matrix");
  if(!(matrixfile.is_open())){
    cout<<"Error: file not found"<<endl;
    return 0;
  }

  for(int i = 0; i < M; i++){
    for(int j =0; j < N; j++){
      matrixfile >> U_t[i*N+j];
	  Uu[i][j] = U_t[i*N+j];
    }
  }
  matrixfile.close();
 
  for(int i=0; i<M;i++){
    for(int j=0; j<N;j++){
      if(i==j){
        V_t[i][j] = 1.0;
      }
      else{
        V_t[i][j] = 0.0;
      }
    }
  }
   //Store A for debug purpose
   for(int i=0; i<M;i++){
      for(int j=0; j<N;j++){
       A[i][j] = U_t[j*N+i];
	   //A[i][j] = U_t[j][i];
      }
    }
 
  // DECLARE VARIABLES - CUDA PORTION
  int acum = 0;
  double converge, c, s, t, zeta;//, alpha;
  double betas[N], gammas[N];
  
  // create  CUDA event handles for timing purposes
  cudaEvent_t start_event, stop_event;
  float elapsed_time_gpu; //,elapsed_time_seq; 

  // Calculate global vars blksz and gridsz  
  blksz = sqrt(N) +(sqrt(N)/2);
  t = (double)N/(double)blksz;
  grdsz = N/blksz;
  if((t - grdsz) > 0.0)
	  grdsz = (int)t + 1;
  else
	  grdsz = (int)t;

  converge = 1.0;
    
  /********************************* START JACOBI ROTATIONS*******************************/
  gettimeofday(&start, NULL);

  while(converge > epsilon){ 
	   
	   converge = 0.0;
	   acum++;
	  
	  for(int i=1; i<M; i++){
		  //alpha = 0.0;
		   /*for(int j=0; j<N ; j++){
			  //alpha += (uiVec[j] * uiVec[j]);			   
		   }*/	

		  if(converge == 0.0){
			  cout << acum << endl;
		  }
		   // *******MAIN COMPUTATION, CUDA VERSION******//		 
		   CUDA_SAFE_CALL( cudaEventCreate(&start_event) );
		   CUDA_SAFE_CALL( cudaEventCreate(&stop_event) );
		   cudaEventRecord(start_event, 0);   
		   ROTATE_GPU_wrapper(U_t, betas, gammas, N, i); 
		   cudaThreadSynchronize();
		   cudaEventRecord(stop_event, 0);
		   cudaEventSynchronize(stop_event);
		   CUDA_SAFE_CALL(cudaEventElapsedTime(&elapsed_time_gpu,start_event, stop_event))		   
		  //}
		   
		  /* 
		  if(acum==2 && i==(M-1))
		  {		   
		    ofstream mgf;
			mgf.open("matrixGPU");
			//mgf<<"# Created from debug\n# name: Vcpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";
			for(int j=0; j<N; j++){
			  mgf<<j<<"\t"<<betas[j]<<"\t"<<gammas[j]<<"\n";
			  }
			mgf.close();
			converge = 0.0;
		   }*/	
		   /*
		   if(i==100)
			   converge = 0.0;
		   else
			   converge = 1.0;*/
		   
		
			for(int j=0; j<i; j++){

				if(j==(i-1))
				{
				   converge = max(converge, abs(gammas[j])/sqrt(betas[i]*betas[j]));
				}			   
			   
			   zeta = (betas[j] - betas[i]) / (2.0 * gammas[j]);
			   t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
			   c = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
			   s = c*t;
			   for(int k=0; k<N; k++){
				   /*
				   PUT MPI CHUNK DISTRIBUTION IN HERE INSTEAD
				   */

				   
				   t = U_t[i*N+k];				  				   
				   U_t[i*N+k] = c*t - s*U_t[j*N+k];				   
				   U_t[j*N+k] = s*t + c*U_t[j*N+k];

				   t = V_t[i][k];			   
				   V_t[i][k] = c*t - s*V_t[j][k];
				   V_t[j][k] = s*t + c*V_t[j][k];
				   
			   }
		   }
	   }//end i-loop
  }//end while-loop
 

  //can parallelize this too
    for(int i =0; i<M; i++){
    t=0;
    for(int j=0; j<N;j++){
      t = t + pow(U_t[i*N+j],2);
    }
    t = sqrt(t);

    for(int j=0; j<N;j++){
      U_t[i*N+j] = U_t[i*N+j] / t;
      if(i == j){
        S[i] = t;
      }
    }	
  }
  

   gettimeofday(&end, NULL);
 /************************************************************/
  

  //printf("%g\n", epsilon);
   //cout << epsilon << endl;

  /*matrixGPU = fopen("matrixGPU", "w");
  for (int i=0; i<N; i++) {
	  fprintf(matrixGPU, "%g\t%g\n", c[i],s[i]);
  }
  fclose(matrixGPU); */

 /* ofstream mgf;
    //File for Matrix V
    mgf.open("matrixGPU");
    //mgf<<"# Created from debug\n# name: Vcpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";
    for(int i = 0; i<N;i++){
      mgf<<*converge<<"\t"<<c[i]<<"\t"<<s[i]<<"\n";
      }
    mgf.close();
	*/

	//cout << sgn(-543) << "\t" << sgn(521) << endl;

  /*
  matrixGPU = fopen("matrixGPU", "w");
  for (int i=0; i<N*N; i++) {
	  fprintf(matrixGPU, "%f\t", U_t[i]);
	  if(i%(N-1) == 0)
		  fprintf(matrixGPU, "\n");
  }
  fclose(matrixGPU); */

  /*
  CUTBoolean res = cutComparefe( h_a, d_a, N, 0.01);
  if (res == 1) {
    printf("VALID!\n  Sequential Time: %.2f msec\n  Parallel Time: %.2f msec\n Speedup = %.2f\n", elapsed_time_seq, elapsed_time_gpu, elapsed_time_seq/elapsed_time_gpu);
  }
  else printf("INVALID...\n"); 
  */


 /* Develop SVD Using OpenMP */
// fix final result

  for(int i =0; i<M; i++){    
    for(int j =0; j<N; j++){
      U[i][j] = U_t[j*N+i];
      V[i][j] = V_t[j][i];      
    }    
  }
  
  //Output time and iterations
  if(T=="-t" || P =="-t"){
    cout<<"iterations: "<<acum<<endl;
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;
  }

  // Output the matrixes for debug
  if(T== "-p" || P == "-p"){
  cout<<"U"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){
      cout<<U[i][j]<<"  ";
    }
    cout<<endl;
  }

  cout<<endl<<"V"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){
      cout<<V[i][j]<<"  ";
    }
    cout<<endl;
  }

  cout<<endl<<"S"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){
       if(i==j){  cout<<S[i]<<"  ";}	
       else{
	       cout<<"0.0  ";
       }
    }
    cout<<endl;
  }
  }

  //Generate Octave files for debug purpouse
   if(Db == "-d" || T == "-d" || P == "-d"){

	/*
    ofstream Af;
    //file for Matrix A
    Af.open("matrixAcuda"); 
    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Af<<" "<<A[i][j];
      }
      Af<<"\n";
    }    
    Af.close();
	*/

    ofstream Uf;	

    //File for Matrix U
    Uf.open("matrixUcuda");
    Uf<<"# Created from debug\n# name: Ucpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";
    
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Uf<<" "<<U[i][j];
      }
      Uf<<"\n";
    }
    Uf.close();

    ofstream Vf;
    //File for Matrix V
    Vf.open("matrixVcuda");
    Vf<<"# Created from debug\n# name: Vcpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Vf<<" "<<V[i][j];
      }
      Vf<<"\n";
    }    

    Vf.close();

    ofstream Sf;
    //File for Matrix S
    Sf.open("matrixScuda");
    Sf<<"# Created from debug\n# name: Scpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";
    
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        if(i == j){
         Sf<<" "<<S[i];

        }
        else{
          Sf<<" 0.0";
        }
      }
      Sf<<"\n";
    }
    Sf.close();
 }

   delete [] S;
   delete [] U_t;
   //delete [] c;
   //delete [] s;
   for(int i = 0; i<N;i++){
	   delete [] G[i];
	   delete [] A[i];
	   delete [] U[i];
	   delete [] Uu[i];
	   delete [] V[i];
	   //delete [] U_t[i];
	   delete [] V_t[i];
   }
   return 0;
}// end main
