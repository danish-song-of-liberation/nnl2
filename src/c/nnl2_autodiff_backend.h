#ifndef NNL2_AUTODIFF_BACKEND
#define NNL2_AUTODIFF_BACKEND

/** @file nnl2_autodiff_backend.h
 ** @date 2025
 ** @copyright MIT
 *
 * The file contains structures and auxiliary 
 * functions for automatic differentiation
 *
 ** Filepath: nnl2/src/c/nnl2_autodiff_backend.h
 **/
 


/// @{ [nnl2_ad_tensor]
 
// Forward declaration
typedef struct nnl2_ad_tensor nnl2_ad_tensor; 
 
/** @brief
 * Tensor structure with automatic differentiation (AD) support
 */ 
typedef struct nnl2_ad_tensor {
	nnl2_object_type ts_type;				   ///< To separate AD tensors from TS tensors
	nnl2_tensor* data;						   ///< Data of the AD tensor
	nnl2_tensor* grad;  					   ///< Gradient of the AD tensor
	bool requires_grad;  					   ///< A flag that determines whether to count the gradient or not
	bool is_leaf; 							   ///< Is the AD tensor the main one or not
	void (* backward_fn)(nnl2_ad_tensor *);    ///< AD-tensor function for backpropagation
	nnl2_ad_tensor** roots; 			       ///< The roots of a tensor
	size_t num_roots;   					   ///< Number of roots
	bool visited;							   ///< To optimize topological sort
	char* name;  						       ///< Name for debugging
	int8_t magic_number; 					   ///< This is necessary to avoid memory corruption when releasing the tensor
	bool grad_initialized;					   ///< If false, the gradient is either NULL or has uninitialized memory
	nnl2_float32 extra_multiplier;			   ///< For edgy cases such as axpy, scale
	bool extra_bool;						   ///< For edgy cases requiring additional boolean
	void* extra_correspondence;				   ///< For correspondence ops
} nnl2_ad_tensor;

/// @} [nnl2_ad_tensor]



/// @{ [nnl2_ad_mode]

/** @brief 
 * Automatic differentiation operation modes
 */
typedef enum {
	nnl2_ad_reverse_mode, 
	nnl2_ad_numerical_p1_mode,  ///< Minimal computation intensity, low precision
 	nnl2_ad_numerical_p2_mode,  ///< Balanced computation intensity, mid precision
	nnl2_ad_numerical_p3_mode   ///< Maximum computation intensity, high precision
} nnl2_ad_mode;

/// @} [nnl2_ad_mode]



/// @{

/** @brief
 * Global constants for common tensor fill values
 *
 * These constants provide zero and one values for different data types
 * to ensure type-safe tensor initialization and avoid pointer lifetime issues
 */

static const nnl2_float32 GLOBAL_ZERO_F32 = 0.0f;
static const nnl2_float64 GLOBAL_ZERO_F64 = 0.0;
static const nnl2_int32 GLOBAL_ZERO_I32 = 0;

static const nnl2_float32 GLOBAL_ONE_F32 = 1.0f;
static const nnl2_float64 GLOBAL_ONE_F64 = 1.0;
static const nnl2_int32 GLOBAL_ONE_I32 = 1;

/// @}



/// @{

/** @brief
 * Returns a pointer to zero value for the specified data type
 * 
 ** @param dtype 
 * Data type for which to get zero value
 *
 ** @return void* 
 * Pointer to zero value constant, NULL if unsupported type
 */
void* nnl2_get_zero_value(nnl2_tensor_type dtype) {
    switch(dtype) {
        case FLOAT32: return (void*)&GLOBAL_ZERO_F32;
        case FLOAT64: return (void*)&GLOBAL_ZERO_F64;
        case INT32:   return (void*)&GLOBAL_ZERO_I32;
        default:      return NULL;
    }
}

/** @brief
 * Returns a pointer to one value for the specified data type
 * 
 ** @param dtype 
 * Data type for which to get one value
 *
 ** @return void* 
 * Pointer to one value constant, NULL if unsupported type
 */
void* nnl2_get_one_value(nnl2_tensor_type dtype) {
    switch(dtype) {
        case FLOAT32: return (void*)&GLOBAL_ONE_F32;
        case FLOAT64: return (void*)&GLOBAL_ONE_F64;
        case INT32:   return (void*)&GLOBAL_ONE_I32;
        default:      return NULL;
    }
}

/// @}


void nnl2_ad_zero_grad(nnl2_ad_tensor* ad_tensor) {
	void* zero = nnl2_get_zero_value(ad_tensor->data->dtype);
	inplace_fill(ad_tensor->grad, &zero, ad_tensor->data->dtype);
}

#endif /** NNL2_AUTODIFF_BACKEND **/
