#ifndef NNL2_CORE_C
#define NNL2_CORE_C

#ifdef __SSE__ 
	#include <xmmintrin.h>
#endif
    
#ifdef NNL2_AVX256_AVAILABLE
	#include <immintrin.h> 
#endif  	

#ifdef __SSE2__ 
	#include <emmintrin.h> 
#endif        
  
#include <stdlib.h> 
#include <time.h>        
  
#include "nnl2_core.h"          	 
#include "nnl2_ffi_test.h"  
#include "nnl2_tensor_core.h"
#include "nnl2_log.h"    
#include "nnl2_foreign_log.h"  
 
#include "backends_status/nnl2_status.h"  

/// NNL2
     
/** @file nnl2_core.c
 ** @brief Contains a function with system initialization and lisp-wrappers
 ** @copyright MIT License 
 ** @date 2025    
 *
 * File contains the full system initialization 
 * declaration as well as all lisp wrappers 
 * for cffi (or sb-alien) 
 *
 ** Filepath: nnl2/src/c/nnl2_core.c
 ** File: nnl2_core.c
 **      
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2	
 **/     
       
///@{ [initilizing]  

///@{ [subinitializers_declaration]            

/** @brief
 * Registers backend implementations for tensor accessor operations
 *
 ** @details 
 * Functions for viewing, getting, and setting tensor data   
 */
void nnl2_init_accessors();

/** @brief 
 * Registers backend implementations for tensor creation operations
 *
 ** @details 
 * Functions for allocating and initializing new tensors    
 */ 
void nnl2_init_tensor_creating();

/** @brief 
 * Registers standard out-of-place mathematical operations
 *
 ** @details 
 * These operations (e.g., add, sub, mul) take input tensors and return a new result tensor
 */
void nnl2_init_standard();    

/** @brief 
 * Registers standard in-place mathematical operations
 *
 ** @details 
 * These operations (e.g., addinplace, subinplace) modify the target tensor directly
 */ 
void nnl2_init_standard_inplace();

/** @brief 
 * Registers backend implementations for tensor stacking and concatenation
 */
void nnl2_init_stack();
       
/** @brief 
 * Registers in-place activation function operations
 *
 ** @details 
 * These functions (e.g., ReLU, Sigmoid) modify the target tensor directly
 */ 
void nnl2_init_activations_inplace();
 
/** @brief 
 * Registers out-of-place activation function operations
 *
 ** @details 
 * These functions (e.g., ReLU, Sigmoid) return a new result tensor
 */
void nnl2_init_activations(); 

/** @brief 
 * Registers backend implementations for tensor weight initializers
 *
 ** @details 
 * Functions like Xavier and random normal distribution initializers
 */
void nnl2_init_initializers();     

/** @brief 
 * Registers backend implementations for tensor transposition operations
 */
void nnl2_init_transposition();

/** @brief 
 * Registers backend implementations for auxiliary/utility operations
 *   
 ** @details
 * Includes summation, normalization, copying, and data filling  
 */
void nnl2_init_auxiliary();     

/** @brief 
 * Registers in-place correspondence operations
 *
 ** @details 
 * In-place operations that correspond to a standard function (e.g., `add_incf_inplace` for `add`)
 */ 
void nnl2_init_correspondence_inplace();       

/** @brief 
 * Registers out-of-place correspondence operations
 *
 ** @details 
 * Out-of-place operations that correspond to a standard function
 */
void nnl2_init_correspondence();

/** @brief  
 * Registers in-place operations with broadcasting support
 *
 ** @details 
 * In-place operations that can handle tensors of different shapes via broadcasting  
 */
void nnl2_init_broadcasting_inplace();
 
/** @brief 
 * Registers out-of-place operations with broadcasting support
 *
 ** @details 
 * Out-of-place operations that can handle tensors of different shapes via broadcasting
 */
void nnl2_init_broadcasting();    
    
/** @brief 
 * Registers backend implementations for tensor reshaping operations 
 *           
 ** @details 
 * Functions to change the shape or interpretation of a tensor's dimensions without copying data
 */
void nnl2_init_reshaping();
           
///@} [subinitializers_declaration]		
		 
/** @brief
 * Fully initializes the nnl2
 *
 ** @warning
 * This function MUST be called before any other functions
 *
 ** @details
 * The function does:
 *  
 ** Seeds the standard random number generator
 *** Then
 ** Initializes the logging subsystem with default 
 ** settings and debug level
 *** Then
 ** Calls all subsystem initialization functions to 
 ** register computational backends for operation
 *
 ** @code    
 * #include "input_here_pass_to_nnl2"
 * int main() {
 *     nnl2_init_system(); // Initialize the framefork first!
 *	   // ... use the framework ...    
 *	   return 0;
 * }	 
 ** @endcode
 ** 
 ** @note
 * The initialization order of the functions is not important
 *
 ** @see nnl2_init_accessors
 ** @see nnl2_init_tensor_creating
 ** @see nnl2_init_standard   
 ** @see nnl2_init_standard_inplace
 ** @see nnl2_init_stack
 ** @see nnl2_init_activations_inplace
 ** @see nnl2_init_activations
 ** @see nnl2_init_initializers
 ** @see nnl2_init_transposition
 ** @see nnl2_init_auxiliary
 ** @see nnl2_init_correspondence_inplace
 ** @see nnl2_init_correspondence 
 ** @see nnl2_init_broadcasting_inplace   
 ** @see nnl2_init_broadcasting
 ** @see nnl2_init_reshaping   
 **/   
void nnl2_init_system() {       
	// Initialization of random number generator
	srand(time(NULL));               
	                                      
	// Initialization of logger            
	nnl2_log_init(             
		NNL2_LOG_DEFAULT_COLOR,       
		NNL2_LOG_DEFAULT_TIMESTAMPS,                  
		NNL2_LOG_DEFAULT_DEBUG_INFO,          
		NNL2_LOG_LEVEL_DEBUG   
	);    
			          
	// Initialization of all functions having several implementations
	nnl2_init_accessors();	  
	nnl2_init_tensor_creating();
	nnl2_init_standard();     
	nnl2_init_standard_inplace();          
	nnl2_init_stack();    
	nnl2_init_activations_inplace();        
	nnl2_init_activations();        
	nnl2_init_auxiliary();
	nnl2_init_initializers();
	nnl2_init_transposition();   
	nnl2_init_correspondence_inplace();    
	nnl2_init_correspondence();   
	nnl2_init_broadcasting_inplace();
	nnl2_init_broadcasting();    
	nnl2_init_reshaping();  	
}                                               
  
///@{ [subinitializers]     

/** @brief See all doxygen at [subinitializers_declaration] **/
              
void nnl2_init_accessors() {
	EINIT_BACKEND(nnl2_view, nnl2_view_backends, CURRENT_BACKEND(nnl2_view));
	INIT_BACKEND(tref_setter, tref_setter_backends);   
	EINIT_BACKEND(nnl2_tref_getter, nnl2_tref_getter_backends, CURRENT_BACKEND(nnl2_tref_getter));
}	     
       
void nnl2_init_tensor_creating() {
	EINIT_BACKEND(inplace_fill, inplace_fill_backends, CURRENT_BACKEND(inplace_fill));
	EINIT_BACKEND(nnl2_empty, nnl2_empty_backends, CURRENT_BACKEND(nnl2_empty));   
}          
     
void nnl2_init_standard() {
	EINIT_BACKEND(add, add_backends, current_backend(add));                       
	EINIT_BACKEND(sub, sub_backends, current_backend(sub));                
	EINIT_BACKEND(mul, mul_backends, current_backend(mul));      	         
	EINIT_BACKEND(nnl2_div, div_backends, current_backend(div));     
	EINIT_BACKEND(nnl2_pow, pow_backends, current_backend(pow));             
	EINIT_BACKEND(nnl2_exp, exp_backends, current_backend(exp));  
	EINIT_BACKEND(nnl2_logarithm, log_backends, current_backend(log));   
	EINIT_BACKEND(scale, scale_backends, current_backend(scale));   
	EINIT_BACKEND(nnl2_max, max_backends, current_backend(max));         
	EINIT_BACKEND(nnl2_min, min_backends, current_backend(min));     	 
	EINIT_BACKEND(nnl2_abs, abs_backends, current_backend(abs));          
	EINIT_BACKEND(axpy, axpy_backends, current_backend(axpy)); 	  
    EINIT_BACKEND(nnl2_neg, neg_backends, current_backend(neg));
	EINIT_BACKEND(nnl2_sqrt, sqrt_backends, current_backend(sqrt));
}
               
void nnl2_init_standard_inplace() {               
	INIT_BACKEND(sgemminplace, sgemminplace_backends);  
	INIT_BACKEND(i32gemminplace, i32gemminplace_backends);         
	EINIT_BACKEND(dgemminplace, dgemminplace_backends, current_backend(gemm)); 
	EINIT_BACKEND(addinplace, addinplace_backends, current_backend(addinplace));       
	EINIT_BACKEND(subinplace, subinplace_backends, current_backend(subinplace));                     
	EINIT_BACKEND(powinplace, powinplace_backends, current_backend(powinplace));     
	EINIT_BACKEND(expinplace, expinplace_backends, current_backend(expinplace));    
	EINIT_BACKEND(loginplace, loginplace_backends, current_backend(loginplace));     	
	EINIT_BACKEND(scaleinplace, scaleinplace_backends, current_backend(scaleinplace));    
	EINIT_BACKEND(maxinplace, maxinplace_backends, current_backend(maxinplace));     
	EINIT_BACKEND(mininplace, mininplace_backends, current_backend(mininplace));   	
	EINIT_BACKEND(mulinplace, mulinplace_backends, current_backend(mulinplace));     
	EINIT_BACKEND(divinplace, divinplace_backends, current_backend(divinplace));  
	EINIT_BACKEND(absinplace, absinplace_backends, current_backend(absinplace));
	EINIT_BACKEND(axpy_inplace, axpy_inplace_backends, current_backend(axpy_inplace));	
	EINIT_BACKEND(nnl2_neginplace, neginplace_backends, current_backend(neginplace));
	EINIT_BACKEND(nnl2_sqrtinplace, sqrtinplace_backends, current_backend(sqrtinplace));
}                            
                               
void nnl2_init_stack() {
	EINIT_BACKEND(hstack, hstack_backends, current_backend(hstack));        
	EINIT_BACKEND(vstack, vstack_backends, current_backend(vstack));
	EINIT_BACKEND(nnl2_concat, concat_backends, current_backend(concat));           
} 

void nnl2_init_activations_inplace() {
	EINIT_BACKEND(reluinplace, reluinplace_backends, current_backend(reluinplace));  
	EINIT_BACKEND(leakyreluinplace, leakyreluinplace_backends, current_backend(leakyreluinplace));
	EINIT_BACKEND(sigmoidinplace, sigmoidinplace_backends, current_backend(sigmoidinplace)); 
	EINIT_BACKEND(tanhinplace, tanhinplace_backends, current_backend(tanhinplace)); 
} 

void nnl2_init_activations() {
	EINIT_BACKEND(relu, relu_backends, current_backend(relu));       
	EINIT_BACKEND(leakyrelu, leakyrelu_backends, current_backend(leakyrelu));     
	EINIT_BACKEND(sigmoid, sigmoid_backends, current_backend(sigmoid)); 
	EINIT_BACKEND(nnl2_tanh, tanh_backends, current_backend(tanh)); 
}
     
void nnl2_init_initializers() {
    EINIT_BACKEND(randn, randn_backends, current_backend(randn));    
    EINIT_BACKEND(xavier, xavier_backends, current_backend(xavier));    
    EINIT_BACKEND(randn_inplace, randn_inplace_backends, current_backend(randn_inplace));    
	EINIT_BACKEND(xavier_inplace, xavier_inplace_backends, current_backend(xavier_inplace));
}
   
void nnl2_init_transposition() {
	EINIT_BACKEND(transposeinplace, transposeinplace_backends, current_backend(transposeinplace)); 
	EINIT_BACKEND(transpose, transpose_backends, current_backend(transpose));  
	EINIT_BACKEND(nnl2_transposition_inplace, transposition_inplace_backends, current_backend(transposition_inplace)); 
	EINIT_BACKEND(nnl2_transposition, transposition_backends, current_backend(transposition));  
}
    
void nnl2_init_auxiliary() {
	EINIT_BACKEND(nnl2_sum_without_axis, sum_without_axis_backends, current_backend(sum_without_axis));  
	INIT_BACKEND(nnl2_sum_with_axis, sum_with_axis_backends);   
	EINIT_BACKEND(l2norm, l2norm_backends, current_backend(l2norm));      
	EINIT_BACKEND(nnl2_copy, copy_backends, current_backend(copy)); 	
	INIT_BACKEND(fill_tensor_with_data, fill_tensor_with_data_backends);
	EINIT_BACKEND(nnl2_slice, slice_backends, CURRENT_BACKEND(slice));
} 

void nnl2_init_correspondence_inplace() {
	INIT_BACKEND(add_incf_inplace, add_incf_inplace_backends); 
	INIT_BACKEND(sub_decf_inplace, sub_decf_inplace_backends);  
	INIT_BACKEND(mul_mulf_inplace, mul_mulf_inplace_backends);  
	INIT_BACKEND(div_divf_inplace, div_divf_inplace_backends);    
	INIT_BACKEND(pow_powf_inplace, pow_powf_inplace_backends);  
	INIT_BACKEND(max_maxf_inplace, max_maxf_inplace_backends);  
	INIT_BACKEND(min_minf_inplace, min_minf_inplace_backends);      
	INIT_BACKEND(axpf_inplace, axpf_inplace_backends);  
}

void nnl2_init_correspondence() { 
	INIT_BACKEND(add_incf, add_incf_backends);    
	INIT_BACKEND(sub_decf, sub_decf_backends);     
	INIT_BACKEND(mul_mulf, mul_mulf_backends);          
	INIT_BACKEND(div_divf, div_divf_backends);   
	INIT_BACKEND(pow_powf, pow_powf_backends);          
	INIT_BACKEND(max_maxf, max_maxf_backends);      
	INIT_BACKEND(min_minf, min_minf_backends); 
	INIT_BACKEND(axpf, axpf_backends);     	    
}      

void nnl2_init_broadcasting_inplace() {
	INIT_BACKEND(add_broadcasting_inplace, add_broadcasting_inplace_backends);
	INIT_BACKEND(sub_broadcasting_inplace, sub_broadcasting_inplace_backends);
	INIT_BACKEND(mul_broadcasting_inplace, mul_broadcasting_inplace_backends); 
	INIT_BACKEND(div_broadcasting_inplace, div_broadcasting_inplace_backends);
	INIT_BACKEND(pow_broadcasting_inplace, pow_broadcasting_inplace_backends);
	INIT_BACKEND(max_broadcasting_inplace, max_broadcasting_inplace_backends);
	INIT_BACKEND(min_broadcasting_inplace, min_broadcasting_inplace_backends);
	INIT_BACKEND(axpy_broadcasting_inplace, axpy_broadcasting_inplace_backends);
}
         
void nnl2_init_broadcasting() {
	INIT_BACKEND(add_broadcasting, add_broadcasting_backends);
	INIT_BACKEND(sub_broadcasting, sub_broadcasting_backends);
	INIT_BACKEND(mul_broadcasting, mul_broadcasting_backends);
	INIT_BACKEND(div_broadcasting, div_broadcasting_backends); 
	INIT_BACKEND(pow_broadcasting, pow_broadcasting_backends);	 
    INIT_BACKEND(max_broadcasting, max_broadcasting_backends);
	INIT_BACKEND(min_broadcasting, min_broadcasting_backends); 	
	INIT_BACKEND(axpy_broadcasting, axpy_broadcasting_backends); 
}     

void nnl2_init_reshaping() {      
	EINIT_BACKEND(nnl2_reshape, reshape_backends, CURRENT_BACKEND(reshape));  
	EINIT_BACKEND(nnl2_reinterpret, reinterpret_backends, CURRENT_BACKEND(reinterpret)); 
}
     
///@} [subinitializers]   
     
///@} [initilizing]


    
///@{ [lisp_wrappers]

/** @brief 
 * Leaving a doxygen here is a repeat of the documentation
 *
 * See the documentation in the declarations of the 
 * functions themselves (or in their typedef declarations)
 */
 
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
          
Tensor* lisp_call_exp(Tensor* tensor, bool save_type) {  
	return nnl2_exp(tensor, save_type); 
}
  
void lisp_call_loginplace(Tensor* tensor) { 
	loginplace(tensor);       
}    
   
Tensor* lisp_call_log(Tensor* tensor, bool save_type) {     
	return nnl2_logarithm(tensor, save_type);
} 
   
void lisp_call_scaleinplace(Tensor* tensor, float multiplier) {
	scaleinplace(tensor, multiplier);     
}

Tensor* lisp_call_scale(Tensor* tensor, float multiplier, bool save_type) {   
	return scale(tensor, multiplier, save_type); 
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
  
void lisp_call_sqrt_inplace(nnl2_tensor* tensor) {
    nnl2_sqrtinplace(tensor);    
}

nnl2_tensor* lisp_call_sqrt(const nnl2_tensor* tensor) {  
    return nnl2_sqrt(tensor);
} 
 
void lisp_call_leakyreluinplace(Tensor* tensor, float alpha) {  
	leakyreluinplace(tensor, alpha);    
} 
   
Tensor* lisp_call_leakyrelu(Tensor* tensor, float alpha, bool save_type) {
	return leakyrelu(tensor, alpha, save_type);  
}    

void lisp_call_sigmoidinplace(Tensor* tensor, bool approx) {        
	sigmoidinplace(tensor, approx);
}      
 
Tensor* lisp_call_sigmoid(Tensor* tensor, bool approx) {  
	return sigmoid(tensor, approx); 
}  
    
void lisp_call_tanhinplace(Tensor* tensor, bool approx) { 
	tanhinplace(tensor, approx);
}    

Tensor* lisp_call_tanh(Tensor* tensor, bool approx) {   
	return nnl2_tanh(tensor, approx); 
}   

Tensor* lisp_call_concat(Tensor* tensora, Tensor* tensorb, int axis) {
	return nnl2_concat(tensora, tensorb, axis); 
}        

Tensor* lisp_call_randn(int* shape, int rank, TensorType dtype, void* from, void* to) {
	return randn(shape, rank, dtype, from, to);  
}

void lisp_call_randn_inplace(nnl2_tensor* tensor, void* from, void* to) {
	randn_inplace(tensor, from, to);
}

Tensor* lisp_call_xavier(int* shape, int rank, TensorType dtype, int in, int out, float gain, float distribution) {
	return xavier(shape, rank, dtype, in, out, gain, distribution);
}                 

void lisp_call_xavier_inplace(nnl2_tensor* tensor, int in, int out, float gain, float distribution) {
	xavier_inplace(tensor, in, out, gain, distribution);
}       
 
void lisp_call_transposeinplace(Tensor* tensor, bool force) {
	transposeinplace(tensor, force);  
}  

Tensor* lisp_call_transpose(Tensor* tensor, bool force) {
	return transpose(tensor, force);    
}

void lisp_call_sum_without_axis(Tensor* tensor, void* filler) {
	nnl2_sum_without_axis(tensor, filler);
}

void lisp_call_sum_with_axis(Tensor* tensor, int axis, bool keepdim) {
	nnl2_sum_with_axis(tensor, axis, keepdim); 
}

void lisp_call_l2norm(Tensor* tensor, int* axes, int num_axes) {
	l2norm(tensor, axes, num_axes); 
}   

Tensor* lisp_call_copy(Tensor* tensor, TensorType copy_type) {
	return nnl2_copy(tensor, copy_type);    
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

Tensor* lisp_call_reshape(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {  
	return nnl2_reshape(tensor, new_shape, new_shape_len, force); 
}

Tensor* lisp_call_reinterpret(Tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force) {  
	return nnl2_reinterpret(tensor, new_shape, new_shape_len, force); 
}

Tensor* lisp_call_slice(Tensor* tensor, int32_t* slice_from, int32_t* slice_to) {
	return nnl2_slice(tensor, slice_from, slice_to);          
}
          
Tensor* lisp_call_transposition(const Tensor* tensor) {
	return nnl2_transposition(tensor);
}
 
void lisp_call_transposition_inplace(Tensor* tensor) {
	nnl2_transposition_inplace(tensor); 
}
    
bool lisp_call_inplace_fill(Tensor* tensor, void* value, TensorType dtype) { 
	return inplace_fill(tensor, value, dtype);
}

void lisp_call_neg_inplace(nnl2_tensor* tensor) {
	nnl2_neginplace(tensor);
}

nnl2_tensor* lisp_call_neg(nnl2_tensor* tensor) { 
    return nnl2_neg(tensor);
}

///@} [lisp_wrappers]             
		             
#endif /** NNL2_CORE_C **/					 
							                    