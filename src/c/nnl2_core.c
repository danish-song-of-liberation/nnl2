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
 
void debug_implementation(Implementation* implementation, char* name, size_t size) {
	printf("Implementation: %s\n", name);
	
	for(size_t i = 0; i < size; i++) {
		printf("	Backend: %s\n", implementation[i].name);
		printf("		Speed: %d\n", implementation[i].speed_priority);
		printf("		Availble?: %d\n", implementation[i].available);
	}
}	
  