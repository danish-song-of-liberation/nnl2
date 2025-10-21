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
	nnl2_tensor* data;						   ///< Data of the AD tensor
	nnl2_tensor* grad;  					   ///< Gradient of the AD tensor
	bool requires_grad;  					   ///< A flag that determines whether to count the gradient or not
	bool is_leaf; 							   ///< Is the AD tensor the main one or not
	void (* backward_fn)(nnl2_ad_tensor *);    ///< AD-tensor function for backpropagation
	nnl2_ad_tensor** roots; 			       ///< The roots of a tensor
	size_t num_roots;   					   ///< Number of roots
	void (* grad_computed)(nnl2_ad_tensor *);  ///< Contains either NULL or the original backpropagation function
	char* name;  						       ///< Name for debugging
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

#endif /** NNL2_AUTODIFF_BACKEND **/
