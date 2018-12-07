#include<stdio.h>
#include<omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.1416
#define N 1000

//Solución del punto 1 del exámen final.
void modelo(double x);
void metropolis (double * lista);
void  exportarDatos(double* x, int num);
int main(int argc, char ** argv)
{
	#pragma omp parallel
	{
	int thread_id = omp_get_thread_num();
	double *x;
	x= malloc(N*sizeof(double));
	metropolis(x);
	exportarDatos(x,thread_id);
	}
	
//	#pragma omp parallel
//	{
	//int thread_id = omp_get_thread_num();
	//int thread_count = omp_get_num_threads();
	//printf("Hello from thread number: %d out of: %d\n",
	//thread_id, thread_count);
	//}
	return 0;
}


//Calcula el la fdp de una de una normal estadar
double f (double x)
{

	float tem = pow(x,2);
	tem = tem/2;
	tem= exp(-tem);
	tem = tem/sqrt(2*PI);
	
	return tem;
}

void metropolis (double * lista)
{	
		lista[0] = drand48();
    
    int i;
    for(i=0; i<N;i++)
    {
    	double delta = drand48()*2 -1;
	    double propuesta = lista[i-1] + delta;
	     
	  	double r = 1.0;
	  	double radio=f(propuesta)/f(lista[i-1]);
	  	if( radio>1)
	  	{
	  		r = radio;
	  	}
	  	
      double alpha=drand48();
      if(alpha<r)
      {    
				lista[i]= propuesta;
      }
      else{
     	 lista[i]=lista[i-1];
     	 }  
    	}
}

void exportarDatos(double* x, int num)
{

	char nombre[10];
	sprintf(nombre, "normal%i.txt", num);


	FILE *arch;
	arch= fopen(nombre, "w");
	if (!arch)
	{
		printf("Problemas abriendo el archivos %s\n", nombre);
		exit(1);
	}

	int	i;
	for (i = 0; i < N; ++i)
	{		
		fprintf(arch, "%f\n",  x[i]);
	}

	fclose(arch);
}
