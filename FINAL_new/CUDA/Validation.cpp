#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#define epsilon 1.e-7

using namespace std;

int main(int argc, char* argv[]){

	double **A, **Ucpu, **Vcpu, **Scpu, **Ucuda, **Vcuda, **Scuda;

	double rUcpu, rVcpu, rScpu, rUcuda, rVcuda, rScuda, resU,resV,resS;
	string dummy;
	int N,M;

	string P; 


	if(argc>1){
		P = argv[1];
	}
	ifstream matrixUcpu("matrixUcpu");
	if(!(matrixUcpu.is_open())){
		cout<<"Error: matrixUcpu, file not found"<<endl;
		return 0;
	}

	ifstream matrixVcpu("matrixVcpu");
	if(!(matrixVcpu.is_open())){
		cout<<"Error: matrixVcpu, file not found"<<endl;
		return 0;
	}

	ifstream matrixScpu("matrixScpu");
	if(!(matrixScpu.is_open())){
		cout<<"Error: matrixScpu, file not found"<<endl;
		return 0;
	}
	ifstream matrixUcuda("matrixUcuda");
	if(!(matrixUcuda.is_open())){
		cout<<"Error: matrixUcuda, file not found"<<endl;
		return 0;
	}

	ifstream matrixVcuda("matrixVcuda");
	if(!(matrixVcuda.is_open())){
		cout<<"Error: matrixVcuda, file not found"<<endl;
		return 0;
	}

	ifstream matrixScuda("matrixScuda");
	if(!(matrixScuda.is_open())){
		cout<<"Error: matrixScuda, file not found"<<endl;
		return 0;
	}

	for( int i = 0; i<12; i++){
		matrixUcpu >> dummy;
		//cout<<dummy;
		matrixVcpu >> dummy;
		matrixScpu >> dummy;
		
		matrixUcuda >> dummy;
		matrixVcuda >> dummy;
		matrixScuda >> dummy;

	}

	matrixUcpu>>M;
	N = M;
	matrixVcpu >> dummy;
	matrixScpu >> dummy;
		
	matrixUcuda >> dummy;
	matrixVcuda >> dummy;
	matrixScuda >> dummy;
	
	for( int i = 0; i<3; i++){
		matrixUcpu >> dummy;
		matrixVcpu >> dummy;
		matrixScpu >> dummy;
		//cout<<dummy;
		
		matrixUcuda >> dummy;
		matrixVcuda >> dummy;
		matrixScuda >> dummy;
	}



	Ucpu = new double *[N];
	Vcpu = new double *[N];
	Scpu = new double *[N];

	Ucuda = new double *[N];
	Vcuda = new double *[N];
	Scuda = new double *[N];

	for(int i =0;i<N;i++){
		Ucpu[i] = new double[N];
		Vcpu[i] = new double[N];
		Scpu[i] = new double[N];

		Ucuda[i] = new double[N];
		Vcuda[i] = new double[N];
		Scuda[i] = new double[N];
	}

	for(int i =0; i<M;i++){
		for(int j =0; j<N;j++){

			matrixUcpu>>Ucpu[i][j];
			
			matrixVcpu>>Vcpu[i][j];
			matrixScpu>>Scpu[i][j];

			matrixUcuda>>Ucuda[i][j];
			matrixVcuda>>Vcuda[i][j];
			matrixScuda>>Scuda[i][j];
		}
	}

	matrixUcpu.close();
	matrixVcpu.close();
	matrixScpu.close();

	matrixUcuda.close();
	matrixVcuda.close();
	matrixScuda.close();




	rUcpu = 0.0;
	rVcpu = 0.0;
	rScpu =0.0;
	rUcuda = 0.0;
	rVcuda = 0.0;
	rScuda = 0.0;

	for(int i = 0; i<N; i++){
		for(int j =0; j<N;j++){

			rUcpu += (abs(Ucpu[i][j]));
			rVcpu += (abs(Vcpu[i][j]));
			rScpu += (abs(Scpu[i][j]));

			rUcuda += (abs(Ucuda[i][j]));
			rVcuda += (abs(Vcuda[i][j]));
			rScuda += (abs(Scuda[i][j]));
		}
	}
	resU = abs(rUcpu - rUcuda);
	resV = abs(rVcpu - rVcuda);
	resS = abs(rScpu - rScuda);

	if((resU >= 0.0 && resU<=epsilon) && (resV >= 0.0 && resV <= epsilon) && (resS >= 0.0 && resS <= epsilon)){
		cout<<"VALID!"<<endl<<endl;
	}

	else{
		cout<<"NOT VALID!"<<endl<<endl;
	}

	if(P == "-p"){

		cout<<"difference in U: "<<resU<<endl<<"difference in V: "<<resV<<endl<<"difference in S: "<<resS<<endl;
	}

	return 0;
}

