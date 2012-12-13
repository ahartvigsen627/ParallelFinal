//Jeremy Bonnell
//CS4230
//Programming Assignment #4

//SVD - MPI IMPLEMENTATION -  BLOCK DISTRIBUTION

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

#include <mpi.h>

#define epsilon 1.e-8

using namespace std;

template <typename T> double sgn(T val)
{
    return (val > T(0)) - (val < T(0));
}

int main (int argc, char* argv[]){

  int M,N;
 
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
  
  double **U,**V, *S, **U_t, **V_t, *A, *Z;
  double *cc, *ss;
  double alpha, zeta, t,t1, acum, c,s;
    
  U = new double*[N];
  V = new double*[N];
  U_t = new double*[N];
  V_t = new double*[N];
  A = new double[N*N];
  Z = new double[N*N];
  S = new double[N];
  cc = new double[N];
  ss = new double[N];
 
  
  for(int i =0; i<N; i++){
	U[i] = new double[N];
 	V[i] = new double[N];
	U_t[i] = new double[N];
	V_t[i] = new double[N];
	//A[i] = new double[N];
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

       A[i*N+j] = U_t[j][i];
      }
    }
   

   for(int i=0; i<M;i++){
      for(int j=0; j<N;j++){

       Z[i*N+j] = V_t[j][i];
      }
    }
  

   double conv = 1.0;
   int my_rank, comm_sz, psize, chunksize, mystart, myend;
   int numberOfThreads = 4;   
   
   double local_b[N];
   double local_g[N];
   double b[N];
   double g[N];
   double local_converge;// = 0.0;
  
   /* SVD using Jacobi algorithm (Parallel)*/
  /************************************************************/
  
   gettimeofday(&start, NULL);

   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   int local_iter;
   int extra_it = 0;
   psize = N/comm_sz;
   mystart=my_rank*psize;
   myend = mystart+psize;
   chunksize = N*psize;

   double local_c[psize];
   double local_s[psize];
   double tempU[chunksize];
   double tempV[chunksize];
   
   for(int i=0; i<psize; i++){
	   local_c[i] = 0.0;
   }
   for(int i=0; i<psize; i++){
	  local_s[i] = 0.0;
   }
   for(int i=mystart; i<myend; i++){
	   int index = i-mystart;
	   for(int j=0; j<N;j++){
		   tempU[index*N+j] = A[i*N+j];
	   }
   }
   for(int i=mystart; i<myend; i++){
	   int index = i-mystart;
	   for(int j=0; j<N;j++){
		   tempV[index*N+j] = Z[i*N+j];
	   }
   }    
  
   while(conv > epsilon){ 		//convergence
	   
	   conv = 0.0;
	   acum++;				//counter of loops	
	   	   
	   for(int i=1; i<M; i++){ 		//convergence

		   //initialize beta/alpha and gamma arrays
		   for(int k=0; k<N; k++){
			   local_b[k] = 0.0;
		   }
		   for(int k=0; k<N; k++){
			  local_g[k] = 0.0;
		   }
		   for(int k=0; k<N; k++){
			   b[k] = 0.0;
		   }
		   for(int k=0; k<N; k++){
			   g[k] = 0.0;
		   }

		   //A is transpose of U_t. Is stored in tempU as chunks of A for each process
			for(int j=0; j<psize; j++){
				//sum my portion of betas and gammas. beta includes alpha when j==i
			   for(int k=0; k<=i; k++){
				   local_b[k] += tempU[j*N+k] * tempU[j*N+k];
				   local_g[k] += tempU[j*N+i] * tempU[j*N+k];
			   }
			}

			MPI_Allreduce(local_b, b, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
			MPI_Allreduce(local_g, g, N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


			// Each process calculates their own c[], s[], and converge
			alpha = b[i];
			conv = max(conv, abs(g[i-1])/sqrt(alpha*b[i-1]));	

			for(int j=0; j<i; j++){

				zeta = (b[j] - alpha) / (2.0 * g[j]);
				t = sgn(zeta) / (abs(zeta) + sqrt(1.0 + (zeta*zeta)));
				cc[j] = 1.0 / (sqrt (1.0 + (t*t)));				
				ss[j] = cc[j]*t;
			}

			//Each process rotates their chunk of A. (A is transpose of U_t)
			for(int j=0; j<psize; j++){
			   for(int k=0; k<i; k++){

				   t = tempU[j*N+i]; 

				   tempU[j*N+i] = cc[k]*t - ss[k]*tempU[j*N+k];
				   tempU[j*N+k] = ss[k]*t + cc[k]*tempU[j*N+k];

				   t1 = tempV[j*N+i];

				   tempV[j*N+i] = cc[k]*t1 - ss[k]*tempV[j*N+k];
				   tempV[j*N+k] = ss[k]*t1 + cc[k]*tempV[j*N+k];
				   
			   }			
			}			
	   }//end i-loop	  
	   
   }//end while

   // FINALLY gather all the chunks back into A and Z. May only need MPI_Gather.
   MPI_Allgather(tempU, chunksize, MPI_DOUBLE, A, chunksize, MPI_DOUBLE, MPI_COMM_WORLD);
   MPI_Allgather(tempV, chunksize, MPI_DOUBLE, Z, chunksize, MPI_DOUBLE, MPI_COMM_WORLD);
   
if(my_rank == 0){

	//Process 0 transposes A back into U_t and Z back into V_t
	for(int j=0; j<M;j++){
		for(int k=0; k<N;k++){
			U_t[j][k] = A[k*N+j];
		}
	}
	for(int j=0; j<M;j++){
		for(int k=0; k<N;k++){
			V_t[j][k] = Z[k*N+j];
		}
	}
	   
  //Create matrix S
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
        Af<<" "<<A[i*N+j];
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
   delete [] A;
   delete [] Z;
   for(int i = 0; i<N;i++){
	   //delete [] A[i];
	   delete [] U[i];
	   delete [] V[i];
	   delete [] U_t[i];
	   delete [] V_t[i];
   }
    MPI_Finalize();
  return 0;
}






