#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>
#include <iostream> 
#include <cmath>
#include "caffe/float_newcublas.cu"
#include "caffe/double_newcublas.cu"
//#include "caffe/float_newcublas.c"
#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

//cancel codes are testing codes

template <>
void caffe_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {//printf("lack transpose GEMM\n");
  
 //if(TransA==CblasNoTrans&&TransB!=CblasNoTrans)printf ("%d----- %d---- %d -----%d ----%d -----%d\n ",CblasNoTrans,TransA,TransB,M,N,K);
  // Note that cublas follows fortran order.
 //if(TransA != CblasNoTrans || TransB != CblasNoTrans)printf("need transpose\n");
  //M N K have questions

 //printf("  ~~~~ M%d    N%d    k%d  \n",M,N,K);
// int lda = (TransA == CblasNoTrans) ? K : M;
 //int ldb = (TransB == CblasNoTrans) ? N : K;
//////////////////


 float *A_trans;
 float *B_trans;
 cudaMalloc((void **)&A_trans, sizeof(float) *K*M);
 cudaMalloc((void **)&B_trans, sizeof(float) *K*N);

// float *A_test;
// float *B_test;

//A_test = (float*)malloc(M*K*sizeof(float));
//B_test = (float*)malloc(K*N*sizeof(float));
//cudaMemcpy(A_test, A, sizeof(float) *K*M, cudaMemcpyDeviceToHost);
//cudaMemcpy(B_test, B, sizeof(float) *K*N, cudaMemcpyDeviceToHost);


//for(int i=0; i<M*K;i++)printf(" %f  ",A_test[i]);
//printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");






  if(TransA == CblasNoTrans){
    
   // cudaMemcpy(A_test, A, sizeof(float) *K*M, cudaMemcpyDeviceToHost);
   // cudaMemcpy(A_trans, A_test, sizeof(float) *K*M, cudaMemcpyHostToDevice);
   cudaMemcpy(A_trans, A, sizeof(float) *K*M, cudaMemcpyDeviceToDevice);
  }
  else {
    // k=colum m =row
    float_transposeCuda( K, M, A, A_trans);
  }

  if(TransB == CblasNoTrans){
   // cudaMemcpy(B_test, B, sizeof(float) *K*N, cudaMemcpyDeviceToHost);
   // cudaMemcpy(B_trans, B_test, sizeof(float) *K*N, cudaMemcpyHostToDevice);
   cudaMemcpy(B_trans, B, sizeof(float) *K*N, cudaMemcpyDeviceToDevice);

  }
  else{
   float_transposeCuda( N, K, B, B_trans);
  }
//cudaMemcpy(A_test, A_trans, sizeof(float) *K*M, cudaMemcpyDeviceToHost);
//cudaMemcpy(B_test, B_trans, sizeof(float) *K*N, cudaMemcpyDeviceToHost);


//for(int i=0; i<M*K;i++)printf(" %f  ",A_test[i]);
//printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");




//for(int i=0; i<N*K;i++)printf(" %f  ",B_test[i]);
//printf("~~~~~~~~~~~~~~~~~~~\n");


//abort();

float_gemmCuda(M,N,K,alpha,beta, A_trans, B_trans, C);

// float *C_test;
//C_test = (float*)malloc(M*N*sizeof(float));
//cudaMemcpy(C_test, C, sizeof(float) *N*M, cudaMemcpyDeviceToHost);

//float_gemmC(M,N,K,alpha,beta, A_test, B_test, C_test);

//cudaMemcpy(C, C_test, sizeof(float) *N*M, cudaMemcpyHostToDevice);

//cudaMemcpy(C_test, C, sizeof(float) *N*M, cudaMemcpyDeviceToHost);

//abort();

//for(int i=0; i<N*M;i++)printf(" %f  ",C_test[i]);
//printf("~~~~~~~~~~~~~~~~~~\n");
//free(C_test);
//free(B_test);
//free(A_test);

cudaFree(A_trans);
cudaFree(B_trans);

/////////////////////////////



 // cublasOperation_t cuTransA =
 //    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 // cublasOperation_t cuTransB =
  //    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
 // CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
  //    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {//printf("lack transpose\n");
// if(TransA != CblasNoTrans || TransB == CblasNoTrans)printf("need transpose\n");
  // Note that cublas follows fortran order.
 // int lda = (TransA == CblasNoTrans) ? K : M;
 // int ldb = (TransB == CblasNoTrans) ? N : K;
/////////////

 double *A_trans;
 double *B_trans;
 cudaMalloc((void **)&A_trans, sizeof(double) *K*M);
 cudaMalloc((void **)&B_trans, sizeof(double) *K*N);
  if(TransA == CblasNoTrans){
    
    cudaMemcpy(A_trans, A, sizeof(double) *K*M, cudaMemcpyDeviceToDevice);
  }
  else {
    
    double_transposeCuda( K, M, A, A_trans);
  }

  if(TransB == CblasNoTrans){
    cudaMemcpy(B_trans, B, sizeof(double) *K*N, cudaMemcpyDeviceToDevice);

  }
  else{
    double_transposeCuda( N, K, B, B_trans);
  }

double_gemmCuda(M,N,K,alpha,beta, A_trans, B_trans, C);

cudaFree(A_trans);
cudaFree(B_trans);

////////////////
//  cublasOperation_t cuTransA =
//    (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  cublasOperation_t cuTransB =
//    (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
//  CUBLAS_CHECK(cublasDgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
//    N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {//printf("lack transpose GEMV\n");
 //if(TransA != CblasNoTrans )printf("need transpose\n");
 float *A_trans;
 cudaMalloc((void **)&A_trans, sizeof(float) *N*M);
int n,m;
  if(TransA == CblasNoTrans){
   n=N;m=M;
    cudaMemcpy(A_trans, A, sizeof(float) *N*M, cudaMemcpyDeviceToDevice);
  }
  else {
   n=M;m=N;
    float_transposeCuda( N, M, A, A_trans);
  }

  float_gemmvCuda(m,n,alpha,beta, A_trans, x, y);
/*
  float *tmp;
    tmp = (float*)malloc(M*sizeof(float));
  float *A_test;
  float *X_test;
  float *Y_test;
  A_test = (float*)malloc(M*N*sizeof(float));
  X_test = (float*)malloc(M*sizeof(float));
  Y_test = (float*)malloc(M*sizeof(float));
  cudaMemcpy(A_test, A, sizeof(float) *N*M, cudaMemcpyDeviceToHost);
  cudaMemcpy(Y_test, Y, sizeof(float) *M, cudaMemcpyDeviceToHost);
  cudaMemcpy(X_test, X, sizeof(float) *N, cudaMemcpyDeviceToHost);
  float_gemmvC(const int M , const int N , const float ALPHA , const float BETA , const float* A , const float* x , float* y ,float *tmp)

*/

  cudaFree(A_trans);


//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
//      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {//rintf("lack transpose\n");
 //if(TransA != CblasNoTrans )printf("need transpose\n");

////////////////////

 double *A_trans;
 cudaMalloc((void **)&A_trans, sizeof(double) *N*M);
 int m,n;
  if(TransA == CblasNoTrans){
    m=M;n=N;
    cudaMemcpy(A_trans, A, sizeof(double) *N*M, cudaMemcpyDeviceToDevice);
  }
  else {
    m=N;n=M;
    double_transposeCuda( M, N, A, A_trans);
  }

  double_gemmvCuda(m,n,alpha,beta,  A_trans, x, y);

  cudaFree(A_trans);

////////////////
//  cublasOperation_t cuTransA =
//      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
//  CUBLAS_CHECK(cublasDgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
//      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
 
    float_axpyCuda(N , alpha, X, Y);

 //CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}
//ok
template <>
void caffe_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
    double_axpyCuda(N , alpha, X, Y);
 // CUBLAS_CHECK(cublasDaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}
//ok
void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float *X) {
 // CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));


 float_scalxCuda(N, alpha, X);
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double *X) {
  //CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
     double_scalxCuda(N, alpha, X);
}

template <>
void caffe_gpu_scal<float>(const int N, const float alpha, float* X,
                           cudaStream_t str) {printf("lack scal stream\n");abort();
 // cudaStream_t initial_stream;
//  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
//  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
//  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
//  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_scal<double>(const int N, const double alpha, double* X,
                            cudaStream_t str) {printf("lack scal stream\n");abort();
// cudaStream_t initial_stream;
//  CUBLAS_CHECK(cublasGetStream(Caffe::cublas_handle(), &initial_stream));
//  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), str));
//  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), N, &alpha, X, 1));
//  CUBLAS_CHECK(cublasSetStream(Caffe::cublas_handle(), initial_stream));
}

template <>
void caffe_gpu_axpby<float>(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal<float>(N, beta, Y);
  caffe_gpu_axpy<float>(N, alpha, X, Y);
}

template <>
void caffe_gpu_axpby<double>(const int N, const double alpha, const double* X,
    const double beta, double* Y) {
  caffe_gpu_scal<double>(N, beta, Y);
  caffe_gpu_axpy<double>(N, alpha, X, Y);
}

template <>
void caffe_gpu_dot<float>(const int n, const float* x, const float* y,
    float* out) {
 
 // CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
 float *A;
 float *B;
 A = (float*)malloc(n*sizeof(float)); 
 B = (float*)malloc(n*sizeof(float)); 
 cudaMemcpy(A, x, sizeof(float) *n, cudaMemcpyDeviceToHost);
 cudaMemcpy(B, y, sizeof(float) *n, cudaMemcpyDeviceToHost);
  
 float sum;
 for(int i=0;i<n;i++){
  //printf(" x %f   y  %f \n",A[i],B[i]);
    sum+=A[i]*B[i];

  }

 free(A);
 free(B);

 //printf("sum %f\n",sum);

 out[0] = sum;
 
 
}

template <>
void caffe_gpu_dot<double>(const int n, const double* x, const double* y,
    double * out) {

 // CUBLAS_CHECK(cublasDdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));

  //  double_dotCuda(n, x, y, out);
  double *A;
  double *B;
  A = (double*)malloc(n*sizeof(double)); 
  B = (double*)malloc(n*sizeof(double)); 
  cudaMemcpy(A, x, sizeof(double) *n, cudaMemcpyDeviceToHost);
  cudaMemcpy(B, y, sizeof(double) *n, cudaMemcpyDeviceToHost);
  
  double sum;
  for(int i=0;i<n;i++){
  //printf(" x %f   y  %f \n",A[i],B[i]);
    sum+=A[i]*B[i];

  }

  free(A);
  free(B);

 //printf("sum %f\n",sum);

  out[0] = sum;
 

}

template <>
void caffe_gpu_asum<float>(const int n, const float* x, float* y) {
 // CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));

      float_asumC( n, x, y);
}
//ok
template <>
void caffe_gpu_asum<double>(const int n, const double* x, double* y) {
//  CUBLAS_CHECK(cublasDasum(Caffe::cublas_handle(), n, x, 1, y));
    double_asumC( n, x, y);
}
//ok
template <>
void caffe_gpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  //CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
 // CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
    float_scalyCuda( n, alpha, x, y);
}

template <>
void caffe_gpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
//  CUBLAS_CHECK(cublasDcopy(Caffe::cublas_handle(), n, x, 1, y, 1));
//  CUBLAS_CHECK(cublasDscal(Caffe::cublas_handle(), n, &alpha, y, 1));

    double_scalyCuda( n, alpha, x, y);
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void log_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}

template <>
void caffe_gpu_log<float>(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <>
void caffe_gpu_log<double>(const int N, const double* a, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, alpha, y);
}

DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index])
                                      - (x[index] < Dtype(0)));
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}

template <>
void caffe_gpu_rng_uniform<float>(const int n, const float a, const float b,
                                  float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_uniform<double>(const int n, const double a, const double b,
                                   double* r) {
  CURAND_CHECK(curandGenerateUniformDouble(Caffe::curand_generator(), r, n));
  const double range = b - a;
  if (range != static_cast<double>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<double>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}

template <>
void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}

template <>
void caffe_gpu_rng_gaussian(const int n, const double mu, const double sigma,
                            double* r) {
  CURAND_CHECK(
      curandGenerateNormalDouble(Caffe::curand_generator(), r, n, mu, sigma));
}

}  // namespace caffe
