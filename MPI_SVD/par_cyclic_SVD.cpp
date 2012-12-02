//Jeremy Bonnell
//CS4230
//Programming Assignment #4

//SVD - MPI and OPENMP IMPLEMENTATION - CYCLIC DISTRIBUTION

/************************************************************************************************/
/* SVD Using Jacobis Rotations									*/
/*												*/
/* Multithreaded								*/
/* Once the “matrix” file is generated compile SVD.cpp: CC –O3 SVD.cpp –o SVD			*/
/* (remember to use CC as capital letter for C++, cc is for C and –xopenmp is the flag for OpenMP)	*/
/* compile SVD.cpp: CC –O3 -xopenmp SVD.cpp –o SVD */
/*												*/
/*												*/
/* Sequential									*/
/* Compile: g++ -O3 SVD.cpp -o SVD								*/
/* Arguments:											*/
/*												*/
/*	M = # of columns									*/
/*	N = # of Rows										*/
/*												*/
/*	Matrix must be squared (M=N)								*/
/*												*/
/*	-t = print out Timing and # of Iterations						*/
/*	-p = print out Results (U, S, V)							*/
/*	-d = Generate the Octave files for debug and verify correctness				*/
/*												*/
/* Use:	./SVD M N -t -p -d									*/
/*												*/
/* All arguments aren't important, just M and N. If you want, is possible to do 		*/
/* ./SVD M N -t and only print out the timing. As well you can use ./SVD M N -d for debug.	*/
/************************************************************************************************/

#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <sys/time.h>

#include <omp.h>
#include <mpi.h>

#define epsilon 1.e-8

using namespace std;

template <typename T> double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

int main (int argc, char* argv[]){

  int M,N;
  //////////////////////////////NUMBER OF THREADS FOR OPENMP CONSTRUCTS!!!!!!!!!/////////////////////
  

  string T,P,Db;
  M = atoi(argv[1]);
  N = atoi(argv[2]);

  double elapsedTime,elapsedTime2;
  timeval start,end,end2;

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
 // cout<<T<<P<<endl;
  
  double **U,**V, *S, **U_t, **V_t, **A;//, *Alphas, *Betas, *Gammas;//, *u;
  double *cc, *ss;//, *tempu;//*beta, *gamma, 
  double alpha, zeta, t,sub_zeta, c,s,temp, temp0, beta,gamma;//converge
  //int conv;

  int acum = 0;
  int temp1, temp2;
  //converge = 1.0;
  //conv = 0;

  U = new double*[N];
  V = new double*[N];
  U_t = new double*[N];
  V_t = new double*[N];
  A = new double*[N];
  S = new double[N];
  cc = new double[N];
  ss = new double[N];
  //tempu = new double[N];
  //double u[1024];// = new double[N];
  //double v[1024];

  //beta = new double[N];
  //gamma = new double[N];
  //converge = new double[N];

  
  for(int i =0; i<N; i++){
	U[i] = new double[N];
 	V[i] = new double[N];
	U_t[i] = new double[N];
	V_t[i] = new double[N];
	A[i] = new double[N];
  }

  //Alphas = new double[N];
  //Betas = new double[N];
  //Gammas = new double[N];

  /*
  for(int i =0; i<N; i++){
	
	Alphas[i] = new double[N];
	Betas[i] = new double[N];
	Gammas[i] = new double[N];
   }
   */


  //Read from file matrix, if not available, app quit
  //Already transposed

  ifstream matrixfile("matrix");
  if(!(matrixfile.is_open())){
    cout<<"Error: file not found"<<endl;
    return 0;
  }

  for(int i = 0; i < M; i++){
    for(int j =0; j < N; j++){

      matrixfile >> U_t[i][j];
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

       A[i][j] = U_t[j][i];
      }
    }


  

  
  /* SVD using Jacobi algorithm (Sequencial)*/
  /************************************************************/

   gettimeofday(&start, NULL);

   double conv = 1.0;
   int my_rank, comm_sz, psize, mystart, myend;
   int numberOfThreads = 4;
   
   //double local_a;
   //double local_b;
   //double local_g;
   double local_converge = 0.0;
   //double tempb[N];// 
  
   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   int local_iter;
   psize = N/comm_sz;
   //mystart=my_rank*psize;
    

   double local_c[N];
   double local_s[N];
   //double local_c = 0.0;
   //double local_s = 0.0;
   
  //#pragma omp parallel num_threads(numberOfThreads)
   //{
   while(conv > epsilon){ 		//convergence
	   
	   local_converge = 0.0;
	   conv = 0.0;	   
	   acum++;				//counter of loops		   	   
	   
	   for(int i=1; i<M; i++){ 		//convergence

		   alpha = 0.0;
		   //beta = 0.0;
		   //local_a = 0.0
		   //tempb[i-1] = 0.0; 
		   local_iter = 0;
		   
		   for(int j=0; j<N; j++){
			   local_c[j] = 0.0;
			   local_s[j] = 0.0;
			   //tempb[i-1] += (U_t[i-1][j] * U_t[i-1][j]);
		   }
		   		  
		   //#pragma omp parallel for private(beta, gamma, zeta, t, local_converge) reduction(+:alpha) 
			for(int j=my_rank; j<i; j+=comm_sz){ //mystart; j<myend; j++){ //0; j<i; j++){	
				
			  beta = 0.0;
			  //beta = tempb[j];
			  gamma = 0.0;				  
			  //local_b = 0.0;
			  //local_g = 0.0;

			  //#pragma omp parallel for num_threads(numberOfThreads) reduction(+:alpha) reduction(+:beta) reduction(+:gamma) //num_threads(numberOfThreads)
			  for(int k = 0; k<N; k++){		//my_rank; k<M; k+=comm_sz){
				  
				  if(j==my_rank)//mystart)
					  alpha += (U_t[i][k] * U_t[i][k]);
				  //if(i==1)
				  beta += (U_t[j][k] * U_t[j][k]);////[i-1][k] * U_t[i-1][k]);
				  gamma += (U_t[i][k] * U_t[j][k]);
				  
			   }			  
			  
			  /*
				if(j==0)
					MPI_Allreduce(&local_a, &alpha, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
				MPI_Allreduce(&local_b, &beta, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				MPI_Allreduce(&local_g, &gamma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
				*/
			  
				if(j==(i-1)){
					//#pragma omp critical
					local_converge = max(local_converge, abs(gamma)/sqrt(alpha*beta));
				}
							
				zeta = (beta - alpha) / (2.0 * gamma);
				t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));        //compute tan of angle
				local_c[j] = 1.0 / (sqrt (1.0 + (t*t)));				//extract cos
				local_s[j] = local_c[j]*t;

				//local_iter++;
				
				//MPI_Allgather(local_c, psize, MPI_DOUBLE, cc, psize, MPI_DOUBLE, MPI_COMM_WORLD);
				//MPI_Allgather(local_s, psize, MPI_DOUBLE, ss, psize, MPI_DOUBLE, MPI_COMM_WORLD);
								
			}
			
			//may need to MPI_Bcast with rank (i-1)
			MPI_Allreduce(&local_converge, &conv, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
			MPI_Allreduce(local_c, cc, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(local_s, ss, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			
			//#pragma omp parallel for num_threads(numberOfThreads) private(t)
			
			for(int j=0; j<i; j++){////0; j<i; j++){
				//if(my_rank == 0)
				//{
			   //#pragma omp parallel for num_threads(numberOfThreads) private(t)
			   for(int k=0; k<N; k++){

				    t = U_t[i][k];
					U_t[i][k] = cc[j]*t - ss[j]*U_t[j][k];
					U_t[j][k] = ss[j]*t + cc[j]*U_t[j][k];

					t = V_t[i][k];
					V_t[i][k] = cc[j]*t - ss[j]*V_t[j][k];
					V_t[j][k] = ss[j]*t + cc[j]*V_t[j][k];
			   }	
				//}
				
			}
			//MPI_Bcast(*U_t, M*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			//MPI_Bcast(*V_t, M*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			//MPI_Allgather(U_t, M*N, MPI_DOUBLE, cc, psize, MPI_DOUBLE, MPI_COMM_WORLD);
			//MPI_Allgather(local_s, psize, MPI_DOUBLE, ss, psize, MPI_DOUBLE, MPI_COMM_WORLD);
	   }	  
   }//end while
   //}//end pragma  

if(my_rank == 0){
	   
  //Create matrix S

#pragma omp parallel for num_threads(numberOfThreads) private(t)
  for(int i =0; i<M; i++){

    t=0;
    for(int j=0; j<N;j++){
      t = t + pow(U_t[i][j],2);
    }
    t = sqrt(t);

    for(int j=0; j<N;j++){
      U_t[i][j] = U_t[i][j] / t;
      if(i == j){
        S[i] = t;
      }
    }
	
  }
  

  gettimeofday(&end, NULL);
 /************************************************************/

 /* Develop SVD Using OpenMP */



// fix final result

  for(int i =0; i<M; i++){
    
    for(int j =0; j<N; j++){

      U[i][j] = U_t[j][i];
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


    ofstream Af;
    //file for Matrix A
    Af.open("matrixAomp"); 
    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";


    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Af<<" "<<A[i][j];
      }
      Af<<"\n";
    }
    
    Af.close();

    ofstream Uf;

    //File for Matrix U
    Uf.open("matrixUomp");
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
    Vf.open("matrixVomp");
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
    Sf.open("matrixSomp");
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
   }

   delete [] S;
   for(int i = 0; i<N;i++){
	   delete [] A[i];
	   delete [] U[i];
	   delete [] V[i];
	   delete [] U_t[i];
	   delete [] V_t[i];
   }
    MPI_Finalize();
  return 0;
}






