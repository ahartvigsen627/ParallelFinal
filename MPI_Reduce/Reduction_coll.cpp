//Jeremy Bonnell
//CS4230
//Programming Assignment #3

//COLLECTIVE VERSION

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

int main (int argc, char* argv[]){

  int M,N;

  string T,P,Db;
  M = atoi(argv[1]);
  N = atoi(argv[2]);

  double elapsedTime,elapsedTime2;
  timeval start,end,end2;

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
  
  double **U_t;
  double alpha, beta, gamma,**Alphas,**Betas,**Gammas;

  int acum = 0;
  int temp1, temp2;
 

  U_t = new double*[N];
  Alphas = new double*[N];
  Betas = new double*[N];
  Gammas = new double*[N];

  for(int i =0; i<N; i++){
	U_t[i] = new double[N];
	Alphas[i] = new double[N];
	Betas[i] = new double[N];
	Gammas[i] = new double[N];
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



  /* Reductions */
  
   gettimeofday(&start, NULL);
   double conv;
   int my_rank, comm_sz;
   
   double local_a;
   double local_b;
   double local_g;
  
   MPI_Init(NULL, NULL);
   MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
   MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

   for(int i =0; i<M;i++){ 		//convergence

    for(int j = 0; j<M; j++){//my_rank; j<M; j+=comm_sz){

	  if(j==0)
		  alpha =0.0;
      gamma = 0.0;
	  
	  local_a = 0.0;
	  local_g = 0.0;

	 
      for(int k =my_rank; k<M; k+=comm_sz){  //0; k<N; k++){
		  if(j==0)
			  local_a = local_a + (U_t[i][k] * U_t[i][k]);
         local_g = local_g +(U_t[i][k] * U_t[j][k]);
       }
		if(j==0)
			MPI_Reduce(&local_a, &alpha, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);		 
		 MPI_Allreduce(&local_g, &gamma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

		 //unnecessary. can be deduced from alpha
		 //MPI_Allreduce(&local_b, &beta, 1, MPI_DOUBLE, MPI_SUM, 1, MPI_COMM_WORLD);
		 
	  if(my_rank == 0){
			Alphas[i][j] = alpha;
			Betas[j][i] = alpha;
			Gammas[i][j] = gamma;
		}
	}

   }
	
   
   
   if(my_rank == 0)
   {

  gettimeofday(&end, NULL);
  


// fix final result


  //Output time and iterations

  if(T=="-t" || P =="-t"){
    elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
    elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
    cout<<"Time: "<<elapsedTime<<" ms."<<endl<<endl;


  }


  // Output the matrixes for debug
  if(T== "-p" || P == "-p"){
  cout<<"Alphas"<<endl<<endl;
  for(int i =0; i<M; i++){

    for(int j =0; j<N;j++){
  		    
    	cout<<Alphas[i][j]<<"  ";
    }
    cout<<endl;
  }

  cout<<endl<<"Betas"<<endl<<endl;
  for(int i =0; i<M; i++){

   for(int j=0; j<N;j++){	  
      cout<<Betas[i][j]<<"  ";
   }
   cout<<endl;
  }

  cout<<endl<<"Gammas"<<endl<<endl;
  for(int i =0; i<M; i++){
    for(int j =0; j<N; j++){

       cout<<Gammas[i][j]<<"  ";
	
     }
    cout<<endl;
  }

  }

  //Generate files for debug purpouse
   if(Db == "-d" || T == "-d" || P == "-d"){


    ofstream Af;
    //file for Matrix A
    Af.open("AlphasMPI.mat"); 
/*    Af<<"# Created from debug\n# name: A\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";

    Af<<M<<"  "<<N;*/
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Af<<" "<<Alphas[i][j];
      }
      Af<<"\n";
    }
    
    Af.close();

    ofstream Uf;

    //File for Matrix U
    Uf.open("BetasMPI.mat");
/*    Uf<<"# Created from debug\n# name: Ugpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";*/
    
    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Uf<<" "<<Betas[i][j];
      }
      Uf<<"\n";
    }
    Uf.close();

    ofstream Vf;
    //File for Matrix V
    Vf.open("GammasMPI.mat");
/*    Vf<<"# Created from debug\n# name: Vgpu\n# type: matrix\n# rows: "<<M<<"\n# columns: "<<N<<"\n";*/

    for(int i = 0; i<M;i++){
      for(int j =0; j<N;j++){
        Vf<<" "<<Gammas[i][j];
      }
      Vf<<"\n";
    }
    

    Vf.close();

    ofstream Sf;


 }
   }

   
   
   for(int i = 0; i<N;i++){
	   delete [] Alphas[i];
	   delete [] U_t[i];
	   delete [] Betas[i];
	   delete [] Gammas[i];
	   
   }
   delete [] Alphas;
   delete [] Betas;
   delete [] Gammas;
   delete [] U_t;
   /*delete [] local_a;
   delete [] local_b*/
   //delete [] local_g;

   MPI_Finalize();

  return 0;
}
