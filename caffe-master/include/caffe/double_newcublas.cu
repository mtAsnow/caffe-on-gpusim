/**
 * dot.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>


#define GPU_DEVICE 0




/* Can switch double between double and double */
//typedef double double;









//dot in cuda


__global__ void dot_kernel(const double *a,const double *b, double *c)
{
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x;
	

	//if(i < 512)
        c[i] =  a[i] * b[i] ;
	
}
//ok

void double_dotCuda(const int n, const double* A,const double* B, double* out)
{       GPU_argv_init();
	//printf("double_dotCuda\n");
        double* C; 
        C = (double*)malloc(n*sizeof(double)); 
        

	
	double *C_gpu;

	
	cudaMalloc((void **)&C_gpu, sizeof(double) *n);
	
	
	cudaMemcpy(C_gpu, C, sizeof(double) *n, cudaMemcpyHostToDevice);
	
	dim3 block(32,32);
	dim3 grid(n/32,1);



	dot_kernel<<< grid, block >>>(A, B, C_gpu);
	cudaThreadSynchronize();




	cudaMemcpy(C, C_gpu, sizeof(double)*n , cudaMemcpyDeviceToHost);    
	
        double gpuResult = 0;	
	for(int k=0;k<n;k++){
           gpuResult+=C[k];
	   //printf(" c  %f   sum   %f  \n", C[k], gpuResult);
	}
        out[0] = gpuResult;

	
	//cudaFree(C_gpu);
      //  free(C); 
}
//ok	


//asum...........in C




void double_asumC(const int n, const double* A, double* out){
    // printf("double_asumC\n");
     double* C; 
     C = (double*)malloc(n*sizeof(double));
     cudaMemcpy(C, A, sizeof(double)*n , cudaMemcpyDeviceToHost);
    

 double   gpuResult = 0;	
 int k = 0;

	for(k=0; k < n ;k++ )
        {
           
            gpuResult+=C[k];
         //  printf("~~~%f ~~~%f \n",gpuResult,C[k]);
	 
	}
        
        *out =  gpuResult;
free(C);



}



//axpy in cuda


__global__ void axpy_kernel(const int n,const double alpha,const double *a, double *b)
{
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x;
	

	if(i < n)b[i] = alpha * a[i] + b[i];
	
}
//ok



void double_axpyCuda(const int n,const double alpha,const double* A, double* C)
{
	
      // printf("double_axpycuda\n");
	
	
	//double *C_gpu;

	
	
	//cudaMalloc((void **)&C_gpu, sizeof(double) *n);
	
	
	
	//cudaMemcpy(C_gpu, C, sizeof(double) *n, cudaMemcpyHostToDevice);
	
	dim3 block(32,32);
	dim3 grid((size_t)(ceil( ((double)n) / ((double)32) )), 1 /*(size_t)(ceil((double)n/ ((double)32) ))*/);



	axpy_kernel<<< grid, block >>>(n, alpha, A,  C);
	cudaThreadSynchronize();




	//cudaMemcpy(C, C_gpu, sizeof(double)*n , cudaMemcpyDeviceToHost);    
	
	

	//cudaFree(C_gpu);
}
//ok	





//scal .. in cuda



__global__ void scalx_kernel(const int n,const double alpha, double *a)
{
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x;
	

	if(i < n)a[i] = alpha * a[i];
	
}
//ok

void double_scalxCuda(const int n,const double alpha, double* A)
{
	//printf("double_scalxCuda\n");

	//double *A_gpu;
	
	

	//cudaMalloc((void **)&A_gpu, sizeof(double) *n);
	
	
	
	//cudaMemcpy(A_gpu, A, sizeof(double) *n, cudaMemcpyHostToDevice);
	
	
	
	dim3 block(32,32);
	dim3 grid((size_t)(ceil( ((double)n) / ((double)32) )), 1 /*(size_t)(ceil((double)n/ ((double)32) ))*/);



	scalx_kernel<<< grid, block >>>(n, alpha, A);
	cudaThreadSynchronize();




	//cudaMemcpy(A, A_gpu, sizeof(double)*n , cudaMemcpyDeviceToHost);    
	
	//cudaFree(A_gpu);

	
}
//ok	

//scaly .. in cuda

__global__ void scaly_kernel(const int n,const double alpha,const double *a, double *b)
{
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x;
	

	if(i < n)b[i] = alpha * a[i];
	
}
//ok

void double_scalyCuda(const int n,const double alpha,const double* A, double* B)
{
	//printf("double_scalyCuda\n");

	//double *A_gpu;
	//double *B_gpu;
	

	//cudaMalloc((void **)&A_gpu, sizeof(double) *n);
	//cudaMalloc((void **)&B_gpu, sizeof(double) *n);
	
	
	//cudaMemcpy(A_gpu, A, sizeof(double) *n, cudaMemcpyHostToDevice);
	//cudaMemcpy(B_gpu, B, sizeof(double) *n, cudaMemcpyHostToDevice);
	
	
	dim3 block(32,32);
	dim3 grid((size_t)(ceil( ((double)n) / ((double)32) )), 1 /*(size_t)(ceil((double)n/ ((double)32) ))*/);



	scaly_kernel<<< grid, block >>>(n, alpha, A, B);
	cudaThreadSynchronize();




	//cudaMemcpy(A, A_gpu, sizeof(double)*n , cudaMemcpyDeviceToHost);    
	//cudaMemcpy(B, B_gpu, sizeof(double)*n , cudaMemcpyDeviceToHost);
	//cudaFree(A_gpu);
	//cudaFree(B_gpu);
	
}

//TRANSPOSE... IN CUDA

__global__ void transpose_kernel(const int m,const int n,const double *a,double *b)
{       int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	

	if((i < n) && (j < m)) b[i*m+j]  =  a[j*n+i];
       // b[i*m+j]  =  a[i*m+j];
	
}
//ok

void double_transposeCuda(const int m,const int n,const double* A, double* C)
{
	

	
	
	dim3 block(32,32);
        dim3 grid((size_t)(ceil( ((double)n) / ((double)32) )), /*1 */(size_t)(ceil((double)m/ ((double)32) )));



	transpose_kernel<<< grid, block >>>(m,n,  A,  C);
	cudaThreadSynchronize();




	
}
//ok	
__global__ void gemm_kernel(const int NI , const int NJ , const int NK , const double ALPHA , const double BETA , double *a , double *b , double *c )
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < NI) && (j < NJ))
	{	
		c[i * NJ + j] *= BETA;
		int k;
		for(k=0; k < NK; k++)
		{
			c[i * NJ + j] += ALPHA * a[i * NK + k] * b[k * NJ +j];
		}
	}
}


void double_gemmCuda(const int NI,const int NJ,const int NK,const double ALPHA, const double BETA,  double* A, double* B, double* C)
{
	


	
	dim3 block(32, 32);
	dim3 grid((size_t)(ceil( ((double)NI) / ((double)1) )),  (size_t)(ceil((double)NJ/ ((double)1) )));
      //dim3 grid(576,576);  


	gemm_kernel<<< grid, block >>>(NI ,NJ ,NK ,ALPHA ,BETA ,A , B, C);
	cudaThreadSynchronize();

	
}
	

//gemm in c for test


void double_gemmC(const int NI,const int NJ,const int NK,const double ALPHA, const double BETA,  double* A, double* B, double* C)
{
	int i,j,k;
	
	for (i = 0; i < NI; i++)
	{
    	for (j = 0; j < NJ; j++)
    	{
			C[i*NJ + j] *= BETA;
	
			for (k = 0; k < NK; ++k)
			{
	  			C[i*NJ + j] += ALPHA * A[i*NK + k] * B[k*NJ + j];
			}
      	}
	}
}


//gemmv in c for test.


void double_gemmvC(const int M , const int N , const double ALPHA , const double BETA , const double* A , const double* x , double* y ,double *tmp)
{
	int i, j;
	
	for (i = 0; i < N; i++)
	{
		tmp[i] = 0;
		//y[i] = 0;
		for (j = 0; j < N; j++)
		{
			tmp[i] = A[i*N + j] * x[j] + tmp[i];
			//y[i] = B[i*N + j] * x[j] + y[i];
		}
		
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}





// gemmv....  in cuda...






__global__ void gesummv_kernel(const int M , const int N , const double ALPHA , const double BETA , const double *a , const double *x , double *y , double *tmp )
{
	int i = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x;

	if (i < M)
	{ tmp[i] = 0;
		int j;
		for(j = 0; j < N; j++)
		{	
			tmp[i] += a[i * N + j] * x[j];
			//y[i] += b[i * N + j] * x[j];
		}
		y[i] = ALPHA * tmp[i] + BETA * y[i];
	}
}

void double_gemmvCuda(const int M , const int N , const double ALPHA , const double BETA , const double* A , const double* x , double* y )
{
		

	double *tmp;
        cudaMalloc((void **)&tmp, sizeof(double) * M);  


	dim3 block(32, 32);
	dim3 grid((size_t)(ceil( ((double)M*N) / ((double)1) )), 1 /*(size_t)(ceil((double)n/ ((double)32) ))*/);


	gesummv_kernel<<< grid, block>>>(M,N,ALPHA,BETA,A,x, y, tmp);
	cudaThreadSynchronize();

        cudaFree(tmp);
	


}



//int main(){return 0;}
