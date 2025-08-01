#ifdef __SSE__
#include <xmmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif
 
#ifdef __SSE3__ 
#include <pmmintrin.h>   
#endif
  
#ifdef __SSSE3__
#include <tmmintrin.h>
#endif 

#ifdef __SSE4_1__ 
#include <smmintrin.h>
#endif

#ifdef __SSE4_2__
#include <nmmintrin.h> 
#endif
 
#include "nnl2_core.h"
#include "nnl2_ffi_test.h" 
#include "backends_status/nnl2_status.h" 
#include "nnl2_tensor_core.h" 

#include <stdlib.h>
#include <time.h>
 
void init_system() {   
	srand(time(NULL)); 

	init_inplace_fill();
	init_empty();
	init_zeros();
	init_ones();
	init_sgemminplace();
	init_dgemminplace();
	init_sgemm();   
	init_dgemm();
	init_addinplace();
	init_subinplace();  
	init_add();       
	init_sub(); 
	init_mulinplace();
	init_divinplace();            
	init_mul();
	init_div(); 
	init_powinplace(); 
	init_expinplace();
	init_pow(); 
	init_exp(); 
	init_loginplace();
	init_log();  
	init_scaleinplace();
	init_scale();
	init_maxinplace();
	init_mininplace();
	init_max();
	init_min(); 
	init_absinplace();
	init_abs();
	init_hstack();
	init_vstack();
	init_reluinplace(); 
	init_relu(); 
	init_leakyreluinplace();
	init_leakyrelu();
	init_sigmoidinplace();
	init_sigmoid();
	init_tanhinplace();
	init_tanh();
	init_concat();
	init_randn();
	init_xavier(); 
	init_transposeinplace();
	init_transpose();
	init_sum();
	init_l2norm();
}     

Tensor* lisp_call_empty(const int* shape, int rank, TensorType dtype) {
	return empty(shape, rank, dtype);
}
 
Tensor* lisp_call_zeros(const int* shape, int rank, TensorType dtype) {
	return zeros(shape, rank, dtype); 
} 

Tensor* lisp_call_ones(const int* shape, int rank, TensorType dtype) { 
	return ones(shape, rank, dtype);   
}                        
        
Tensor* lisp_call_full(const int* shape, int rank, TensorType dtype, void* filler) {
	return full(shape, rank, dtype, filler);
}
      
Tensor* lisp_call_dgemm(const nnl2_order order, const nnl2_transpose transa, 
						const nnl2_transpose transb, const int m, const int n, 
						const int k, const double alpha, const Tensor* a, const int lda,
						const Tensor* b, const int ldb, const double beta) {    
							    
	return dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta);
}          
   
Tensor* lisp_call_sgemm(const nnl2_order order, const nnl2_transpose transa, 
						const nnl2_transpose transb, const int m, const int n, 
						const int k, const float alpha, const Tensor* a, const int lda,
						const Tensor* b, const int ldb, const float beta) {
							   
	return sgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta);
}          
 
void lisp_call_addinplace(Tensor* summand, Tensor* addend) { 
	addinplace(summand, addend);
}  

void lisp_call_subinplace(Tensor* summand, Tensor* addend) {
	subinplace(summand, addend);
}

Tensor* lisp_call_add(Tensor* summand, Tensor* addend) {
	return add(summand, addend);  
}

Tensor* lisp_call_sub(Tensor* summand, Tensor* addend) {
	return sub(summand, addend); 
}    

void lisp_call_mulinplace(Tensor* multiplicand, Tensor* multiplier) {
	mulinplace(multiplicand, multiplier);
}

void lisp_call_divinplace(Tensor* dividend, Tensor* divisor) {
	divinplace(dividend, divisor);
}

Tensor* lisp_call_mul(Tensor* multiplicand, Tensor* multiplier) {
	return mul(multiplicand, multiplier);    
}

Tensor* lisp_call_div(Tensor* dividend, Tensor* divisor) {
	return nnl2_div(dividend, divisor); 
} 

void lisp_call_powinplace(Tensor* base, Tensor* exponent) {
	powinplace(base, exponent);
} 

void lisp_call_expinplace(Tensor* tensor) {
	expinplace(tensor);
}

Tensor* lisp_call_pow(Tensor* base, Tensor* exponent) {
	return nnl2_pow(base, exponent);
} 
         
Tensor* lisp_call_exp(Tensor* tensor) {
	return nnl2_exp(tensor);
}
 
void lisp_call_loginplace(Tensor* tensor) { 
	loginplace(tensor);      
}
   
Tensor* lisp_call_log(Tensor* tensor) {
	return nnl2_log(tensor);
} 
  
void lisp_call_scaleinplace(Tensor* tensor, float multiplier) {
	scaleinplace(tensor, multiplier);  
}

Tensor* lisp_call_scale(Tensor* tensor, float multiplier) {
	return scale(tensor, multiplier); 
}
 
void lisp_call_maxinplace(Tensor* tensora, Tensor* tensorb) {
	maxinplace(tensora, tensorb);
}

void lisp_call_mininplace(Tensor* tensora, Tensor* tensorb) {
	mininplace(tensora, tensorb);
}

Tensor* lisp_call_max(Tensor* tensora, Tensor* tensorb) {
	return nnl2_max(tensora, tensorb);  
} 
  
Tensor* lisp_call_min(Tensor* tensora, Tensor* tensorb) {
	return nnl2_min(tensora, tensorb);   
}  
  
void lisp_call_absinplace(Tensor* tensor) {
	absinplace(tensor); 
}
  
Tensor* lisp_call_abs(Tensor* tensor) {
	return nnl2_abs(tensor); 
}  

Tensor* lisp_call_hstack(Tensor* tensora, Tensor* tensorb) {
	return hstack(tensora, tensorb);
}  

Tensor* lisp_call_vstack(Tensor* tensora, Tensor* tensorb) {  
	return vstack(tensora, tensorb);
}
 
void lisp_call_reluinplace(Tensor* tensor) {
	reluinplace(tensor);
}  	

Tensor* lisp_call_relu(Tensor* tensor) {
	return relu(tensor);
}
 
void lisp_call_leakyreluinplace(Tensor* tensor, float alpha) {
	leakyreluinplace(tensor, alpha); 
}

Tensor* lisp_call_leakyrelu(Tensor* tensor, float alpha) {
	return leakyrelu(tensor, alpha);  
}

void lisp_call_sigmoidinplace(Tensor* tensor) {
	sigmoidinplace(tensor);
}

Tensor* lisp_call_sigmoid(Tensor* tensor) {
	return sigmoid(tensor);
}
 
void lisp_call_tanhinplace(Tensor* tensor) { 
	tanhinplace(tensor);
}    

Tensor* lisp_call_tanh(Tensor* tensor) {
	return nnl2_tanh(tensor);
}   

Tensor* lisp_call_concat(Tensor* tensora, Tensor* tensorb, int axis) {
	return nnl2_concat(tensora, tensorb, axis);
}        

Tensor* lisp_call_randn(int* shape, int rank, TensorType dtype, void* from, void* to) {
	return randn(shape, rank, dtype, from, to);
}

Tensor* lisp_call_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	return xavier(shape, rank, dtype, in, out, gain, distribution);
}

void lisp_call_transposeinplace(Tensor* tensor) {
	transposeinplace(tensor);  
}

Tensor* lisp_call_transpose(Tensor* tensor) {
	return transpose(tensor);  
}

void lisp_call_sum(Tensor* tensor, int* axes, int num_axes) {
	nnl2_sum(tensor, axes, num_axes);
}

void lisp_call_l2norm(Tensor* tensor, int* axes, int num_axes) {
	l2norm(tensor, axes, num_axes);
}
     
void debug_implementation(Implementation* implementation, char* name, size_t size) {  
	printf("Implementation: %s\n", name);
	
	for(size_t i = 0; i < size; i++) {
		printf("	Backend: %s\n", implementation[i].name);
		printf("		Speed: %d\n", implementation[i].speed_priority);
		printf("		Availble?: %d\n", implementation[i].available);
	} 
}	           

void lisp_call_debug_blas_sgemminplace(size_t check_to) {
	debug_implementation(sgemminplace_backends, "sgemm in place", check_to);
}		
		       