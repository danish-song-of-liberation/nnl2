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
 
void init_system() {
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
    
Tensor* lisp_call_full(const int* shape, int rank, TensorType  dtype, void* filler) {
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
		