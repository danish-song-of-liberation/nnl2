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
#include "nnl2_tensor_core.h" 
#include "nnl2_log.h"

#include "backends_status/nnl2_status.h" 

#include <stdlib.h>
#include <time.h> 
       
void init_system() {     
	srand(time(NULL));             
	     
	// Initialization of the logging system         
	nnl2_log_init(             
		NNL2_LOG_DEFAULT_COLOR,       
		NNL2_LOG_DEFAULT_TIMESTAMPS,            
		NNL2_LOG_DEFAULT_DEBUG_INFO,         
		NNL2_LOG_LEVEL_DEBUG  
	); 
           
	EINIT_BACKEND(nnl2_view, nnl2_view_backends, CURRENT_BACKEND(nnl2_view));
	INIT_BACKEND(tref_setter, tref_setter_backends);   
	EINIT_BACKEND(nnl2_tref_getter, nnl2_tref_getter_backends, CURRENT_BACKEND(nnl2_tref_getter));
	EINIT_BACKEND(inplace_fill, inplace_fill_backends, CURRENT_BACKEND(inplace_fill));
	EINIT_BACKEND(nnl2_empty, nnl2_empty_backends, CURRENT_BACKEND(nnl2_empty));   
	EINIT_BACKEND(nnl2_zeros, nnl2_zeros_backends, CURRENT_BACKEND(nnl2_zeros));   
	INIT_BACKEND(sgemminplace, sgemminplace_backends);   
	EINIT_BACKEND(dgemminplace, dgemminplace_backends, current_backend(gemm)); 
	EINIT_BACKEND(addinplace, addinplace_backends, current_backend(addinplace));       
	EINIT_BACKEND(subinplace, subinplace_backends, current_backend(subinplace));     
	EINIT_BACKEND(add, add_backends, current_backend(add));   
	EINIT_BACKEND(sub, sub_backends, current_backend(sub));      
	EINIT_BACKEND(mulinplace, mulinplace_backends, current_backend(mulinplace));  
	EINIT_BACKEND(divinplace, divinplace_backends, current_backend(divinplace));         
	EINIT_BACKEND(mul, mul_backends, current_backend(mul));      	      
	EINIT_BACKEND(nnl2_div, div_backends, current_backend(div));    
	EINIT_BACKEND(powinplace, powinplace_backends, current_backend(powinplace));     
	EINIT_BACKEND(expinplace, expinplace_backends, current_backend(expinplace));     
	EINIT_BACKEND(nnl2_pow, pow_backends, current_backend(pow));        
	EINIT_BACKEND(nnl2_exp, exp_backends, current_backend(exp)); 
	EINIT_BACKEND(loginplace, loginplace_backends, current_backend(loginplace));  
	EINIT_BACKEND(nnl2_logarithm, log_backends, current_backend(log)); 
	EINIT_BACKEND(scaleinplace, scaleinplace_backends, current_backend(scaleinplace)); 
	EINIT_BACKEND(scale, scale_backends, current_backend(scale));   
	EINIT_BACKEND(maxinplace, maxinplace_backends, current_backend(maxinplace));  
	EINIT_BACKEND(mininplace, mininplace_backends, current_backend(mininplace)); 
	EINIT_BACKEND(nnl2_max, max_backends, current_backend(max));      
	EINIT_BACKEND(nnl2_min, min_backends, current_backend(min));   
	EINIT_BACKEND(absinplace, absinplace_backends, current_backend(absinplace));
	EINIT_BACKEND(nnl2_abs, abs_backends, current_backend(abs));  
	EINIT_BACKEND(hstack, hstack_backends, current_backend(hstack));        
	EINIT_BACKEND(vstack, vstack_backends, current_backend(vstack));
	EINIT_BACKEND(reluinplace, reluinplace_backends, current_backend(reluinplace));       
	EINIT_BACKEND(relu, relu_backends, current_backend(relu));    
	EINIT_BACKEND(leakyreluinplace, leakyreluinplace_backends, current_backend(leakyreluinplace));  
	EINIT_BACKEND(leakyrelu, leakyrelu_backends, current_backend(leakyrelu));   
	EINIT_BACKEND(sigmoidinplace, sigmoidinplace_backends, current_backend(sigmoidinplace)); 
	EINIT_BACKEND(sigmoid, sigmoid_backends, current_backend(sigmoid)); 
	EINIT_BACKEND(tanhinplace, tanhinplace_backends, current_backend(tanhinplace)); 
	EINIT_BACKEND(nnl2_tanh, tanh_backends, current_backend(tanh)); 
	EINIT_BACKEND(nnl2_concat, concat_backends, current_backend(concat));     
	EINIT_BACKEND(randn, randn_backends, current_backend(randn));  
	EINIT_BACKEND(xavier, xavier_backends, current_backend(xavier));  
	EINIT_BACKEND(transposeinplace, transposeinplace_backends, current_backend(transposeinplace)); 
	EINIT_BACKEND(transpose, transpose_backends, current_backend(transpose));  
	EINIT_BACKEND(nnl2_sum, sum_backends, current_backend(sum));    
	EINIT_BACKEND(l2norm, l2norm_backends, current_backend(l2norm));  
	EINIT_BACKEND(nnl2_copy, copy_backends, current_backend(copy)); 	
	INIT_BACKEND(add_incf_inplace, add_incf_inplace_backends); 
	INIT_BACKEND(add_incf, add_incf_backends);    
	INIT_BACKEND(sub_decf_inplace, sub_decf_inplace_backends); 
	INIT_BACKEND(sub_decf, sub_decf_backends); 
	INIT_BACKEND(mul_mulf_inplace, mul_mulf_inplace_backends);  
	INIT_BACKEND(mul_mulf, mul_mulf_backends); 
	INIT_BACKEND(div_divf_inplace, div_divf_inplace_backends);    
	INIT_BACKEND(div_divf, div_divf_backends);  
	INIT_BACKEND(pow_powf_inplace, pow_powf_inplace_backends); 
	INIT_BACKEND(pow_powf, pow_powf_backends);   
	INIT_BACKEND(max_maxf_inplace, max_maxf_inplace_backends);        
	INIT_BACKEND(max_maxf, max_maxf_backends); 
	INIT_BACKEND(min_minf_inplace, min_minf_inplace_backends);   
	INIT_BACKEND(min_minf, min_minf_backends);  
	INIT_BACKEND(add_broadcasting_inplace, add_broadcasting_inplace_backends);
	INIT_BACKEND(add_broadcasting, add_broadcasting_backends);
	INIT_BACKEND(sub_broadcasting_inplace, sub_broadcasting_inplace_backends);
	INIT_BACKEND(sub_broadcasting, sub_broadcasting_backends);
	INIT_BACKEND(mul_broadcasting_inplace, mul_broadcasting_inplace_backends);
	INIT_BACKEND(mul_broadcasting, mul_broadcasting_backends);
	INIT_BACKEND(div_broadcasting_inplace, div_broadcasting_inplace_backends);
	INIT_BACKEND(div_broadcasting, div_broadcasting_backends);
	INIT_BACKEND(pow_broadcasting_inplace, pow_broadcasting_inplace_backends);
	INIT_BACKEND(pow_broadcasting, pow_broadcasting_backends);	 
	INIT_BACKEND(max_broadcasting_inplace, max_broadcasting_inplace_backends);
	INIT_BACKEND(min_broadcasting_inplace, min_broadcasting_inplace_backends);
	INIT_BACKEND(max_broadcasting, max_broadcasting_backends);
	INIT_BACKEND(min_broadcasting, min_broadcasting_backends); 
	INIT_BACKEND(fill_tensor_with_data, fill_tensor_with_data_backends);
	EINIT_BACKEND(axpy_inplace, axpy_inplace_backends, current_backend(axpy_inplace));
	EINIT_BACKEND(axpy, axpy_backends, current_backend(axpy));
	INIT_BACKEND(axpf_inplace, axpf_inplace_backends);
	INIT_BACKEND(axpf, axpf_backends);    
	INIT_BACKEND(axpy_broadcasting_inplace, axpy_broadcasting_inplace_backends);
	INIT_BACKEND(axpy_broadcasting, axpy_broadcasting_backends);   
}                                   

void* lisp_call_view(Tensor* tensor, int32_t* indices, uint8_t num_indices) {
	return nnl2_view(tensor, indices, num_indices);
} 

void lisp_call_tref_setter(Tensor* tensor, int* shape, int rank, void* change_with, bool tensor_p) {
	tref_setter(tensor, shape, rank, change_with, tensor_p);
}

void* lisp_call_tref_getter(Tensor* tensor, int32_t* indices, uint8_t num_indices) {
	return nnl2_tref_getter(tensor, indices, num_indices);
} 
     
Tensor* lisp_call_empty(const int* shape, int rank, TensorType dtype) {
	return nnl2_empty(shape, rank, dtype);
} 
     
Tensor* lisp_call_zeros(const int* shape, int rank, TensorType dtype) {
	return nnl2_zeros(shape, rank, dtype);  
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
	return nnl2_logarithm(tensor);
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

Tensor* lisp_call_copy(Tensor* tensor) {
	return nnl2_copy(tensor); 
} 

void lisp_call_add_incf_inplace(Tensor* tensor, void* inc) {
	add_incf_inplace(tensor, inc);
}

Tensor* lisp_call_add_incf(Tensor* tensor, void* inc) {
	return add_incf(tensor, inc);
}

void lisp_call_sub_decf_inplace(Tensor* tensor, void* dec) {
	sub_decf_inplace(tensor, dec);
}

Tensor* lisp_call_sub_decf(Tensor* tensor, void* dec) {
	return sub_decf(tensor, dec);
}
     
void lisp_call_mul_mulf_inplace(Tensor* tensor, void* mulf) {	
	mul_mulf_inplace(tensor, mulf);
}	

Tensor* lisp_call_mul_mulf(Tensor* tensor, void* mulf) {
	return mul_mulf(tensor, mulf);
}

void lisp_call_div_divf_inplace(Tensor* tensor, void* divf) {
	div_divf_inplace(tensor, divf);
}

Tensor* lisp_call_div_divf(Tensor* tensor, void* divf) {
	return div_divf(tensor, divf);
}

void lisp_call_pow_powf_inplace(Tensor* tensor, void* powf_arg) {
	pow_powf_inplace(tensor, powf_arg);
}
 
Tensor* lisp_call_pow_powf(Tensor* tensor, void* powf) {
	return pow_powf(tensor, powf);
}
	 
void lisp_call_max_maxf_inplace(Tensor* tensor, void* maxf) {
	max_maxf_inplace(tensor, maxf);
}

Tensor* lisp_call_max_maxf(Tensor* tensor, void* maxf) {
	return max_maxf(tensor, maxf);
}
 
void lisp_call_min_minf_inplace(Tensor* tensor, void* minf) {
	min_minf_inplace(tensor, minf);
} 

Tensor* lisp_call_min_minf(Tensor* tensor, void* minf) {
	return min_minf(tensor, minf);
}  
  
void lisp_call_add_broadcasting_inplace(Tensor* summand, Tensor* sumend) {
	return add_broadcasting_inplace(summand, sumend);
}

Tensor* lisp_call_add_broadcasting(Tensor* summand, Tensor* sumend) {
	return add_broadcasting(summand, sumend);  
}    
  
void lisp_call_sub_broadcasting_inplace(Tensor* summand, Tensor* sumend) {
	return sub_broadcasting_inplace(summand, sumend);
} 
 
Tensor* lisp_call_sub_broadcasting(Tensor* minuend, Tensor* subtrahend) { 
	return sub_broadcasting(minuend, subtrahend); 
}

void lisp_call_mul_broadcasting_inplace(Tensor* multiplicand, Tensor* multiplier) {
	return mul_broadcasting_inplace(multiplicand, multiplier);
}

Tensor* lisp_call_mul_broadcasting(Tensor* multiplicand, Tensor* multiplier) { 
	return mul_broadcasting(multiplicand, multiplier);
}

void lisp_call_div_broadcasting_inplace(Tensor* dividend, Tensor* divisor) {
	return div_broadcasting_inplace(dividend, divisor);
}

Tensor* lisp_call_div_broadcasting(Tensor* dividend, Tensor* divisor) { 
	return div_broadcasting(dividend, divisor);
}

void lisp_call_pow_broadcasting_inplace(Tensor* base, Tensor* exponent) {
	return pow_broadcasting_inplace(base, exponent);
}

Tensor* lisp_call_pow_broadcasting(Tensor* base, Tensor* exponent) { 
	return pow_broadcasting(base, exponent);
}
	 
void lisp_call_max_broadcasting_inplace(Tensor* a, Tensor* b) {
	return max_broadcasting_inplace(a, b);
}

void lisp_call_min_broadcasting_inplace(Tensor* a, Tensor* b) {
	return min_broadcasting_inplace(a, b);
}

Tensor* lisp_call_max_broadcasting(Tensor* a, Tensor* b) { 
	return max_broadcasting(a, b);
}

Tensor* lisp_call_min_broadcasting(Tensor* a, Tensor* b) { 
	return min_broadcasting(a, b);
} 

void lisp_call_axpy_inplace(Tensor* summand, Tensor* sumend, float alpha) {
	axpy_inplace(summand, sumend, alpha);    
}

Tensor* lisp_call_axpy(Tensor* summand, Tensor* sumend, float alpha) {
	return axpy(summand, sumend, alpha);    
} 

void lisp_call_axpf_inplace(Tensor* summand, void* sumend, float alpha) {
	axpf_inplace(summand, sumend, alpha); 
} 

Tensor* lisp_call_axpf(Tensor* summand, void* sumend, float alpha) {
	return axpf(summand, sumend, alpha);  
} 

void lisp_call_axpy_broadcasting_inplace(Tensor* summand, Tensor* sumend, float alpha) {
	axpy_broadcasting_inplace(summand, sumend, alpha); 
}   

Tensor* lisp_call_axpy_broadcasting(Tensor* summand, void* sumend, float alpha) {
	return axpy_broadcasting(summand, sumend, alpha); 
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
		             