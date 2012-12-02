#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath> 
#define epsilon 1.e-8

using namespace std;

int main(int argc, char* argv[]){

	double **A, **Ucpu, **Vcpu, **Scpu, **Uomp, **Vomp, **Somp;

	double rUcpu, rVcpu, rScpu, rUomp, rVomp, rSomp, resU,resV,resS;
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
	ifstream matrixUomp("matrixUomp");
	if(!(matrixUomp.is_open())){
		cout<<"Error: matrixUomp, file not found"<<endl;
		return 0;
	}

	ifstream matrixVomp("matrixVomp");
	if(!(matrixVomp.is_open())){
		cout<<"Error: matrixVomp, file not found"<<endl;
		return 0;
	}

	ifstream matrixSomp("matrixSomp");
	if(!(matrixSomp.is_open())){
		cout<<"Error: matrixSomp, file not found"<<endl;
		return 0;
	}

	for( int i = 0; i<12; i++){
		matrixUcpu >> dummy;
		//cout<<dummy;
		matrixVcpu >> dummy;
		matrixScpu >> dummy;
		
		matrixUomp >> dummy;
		matrixVomp >> dummy;
		matrixSomp >> dummy;

	}

	matrixUcpu>>M;
	N = M;
	matrixVcpu >> dummy;
	matrixScpu >> dummy;
		
	matrixUomp >> dummy;
	matrixVomp >> dummy;
	matrixSomp >> dummy;
	
	for( int i = 0; i<3; i++){
		matrixUcpu >> dummy;
		matrixVcpu >> dummy;
		matrixScpu >> dummy;
		//cout<<dummy;
		
		matrixUomp >> dummy;
		matrixVomp >> dummy;
		matrixSomp >> dummy;
	}



	Ucpu = new double *[N];
	Vcpu = new double *[N];
	Scpu = new double *[N];

	Uomp = new double *[N];
	Vomp = new double *[N];
	Somp = new double *[N];

	for(int i =0;i<N;i++){
		Ucpu[i] = new double[N];
		Vcpu[i] = new double[N];
		Scpu[i] = new double[N];

		Uomp[i] = new double[N];
		Vomp[i] = new double[N];
		Somp[i] = new double[N];
	}

	for(int i =0; i<M;i++){
		for(int j =0; j<N;j++){

			matrixUcpu>>Ucpu[i][j];
			
			matrixVcpu>>Vcpu[i][j];
			matrixScpu>>Scpu[i][j];

			matrixUomp>>Uomp[i][j];
			matrixVomp>>Vomp[i][j];
			matrixSomp>>Somp[i][j];
		}
	}

	matrixUcpu.close();
	matrixVcpu.close();
	matrixScpu.close();

	matrixUomp.close();
	matrixVomp.close();
	matrixSomp.close();




	rUcpu = 0.0;
	rVcpu = 0.0;
	rScpu =0.0;
	rUomp = 0.0;
	rVomp = 0.0;
	rSomp = 0.0;

	for(int i = 0; i<N; i++){
		for(int j =0; j<N;j++){

			rUcpu += (abs(Ucpu[i][j]));
			rVcpu += (abs(Vcpu[i][j]));
			rScpu += (abs(Scpu[i][j]));

			rUomp += (abs(Uomp[i][j]));
			rVomp += (abs(Vomp[i][j]));
			rSomp += (abs(Somp[i][j]));
		}
	}
	resU = abs(rUcpu - rUomp);
	resV = abs(rVcpu - rVomp);
	resS = abs(rScpu - rSomp);

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

