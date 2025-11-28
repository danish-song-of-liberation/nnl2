#ifndef NNL2_AD_GEMMVP_H
#define NNL2_AD_GEMMVP_H

/** @file nnl2_ad_gemmvp.h
 ** @brief AD implementation for GEMMVP (General Matrix Multiplication with Vector Product) operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for GEMMVP operation
 *
 ** @param tensor
 * The output tensor from GEMMVP operation that needs gradient computation
 *
 ** @details
 * Derivatives of matrix multiplication with bias: C = A @ B + V
 * dC/dA = gradient @ B^T
 * dC/dB = A^T @ gradient  
 * dC/dV = gradient (summed along appropriate dimensions)
 * Propagates gradients to all three input tensors using matrix calculus rules
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->roots[0], tensor->roots[1], or tensor->roots[2] is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_gemmvp()
 ** @see nnl2_ad_reverse_derivative_gemmvp()
 **/	
static void nnl2_ad_reverse_backward_gemmvp(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_gemmvp, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_gemmvp, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_gemmvp, multiplicand root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[1], "In function nnl2_ad_reverse_backward_gemmvp, multiplier root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[2], "In function nnl2_ad_reverse_backward_gemmvp, bias vector root tensor is NULL");
	#endif
	
	// Compute the derivative of GEMMVP operation and propagate to root tensors
    nnl2_ad_reverse_derivative_gemmvp(tensor, tensor->roots[0], tensor->roots[1], tensor->roots[2]);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for GEMMVP (General Matrix Multiplication with Vector Product) operation
 *
 ** @param multiplicand 
 * First input tensor (matrix A) with shape [M, K]
 *
 ** @param multiplier 
 * Second input tensor (matrix B) with shape [K, N] 
 *
 ** @param vector
 * Bias vector tensor with shape [N] for row-wise addition
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing matrix product A × B + V with shape [M, N], or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if multiplicand, multiplier or vector is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if input tensors have incompatible dimensions
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if gemmvp operation fails on input data
 *
 ** @exception NNL2Error
 * Returns NULL if gradient tensor allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if roots array allocation fails when track_graph=true
 *
 ** @exception NNL2Error
 * Returns NULL if unknown AD mode is specified
 *
 ** @see nnl2_gemmvp()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_gemmvp(nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier, nnl2_ad_tensor* vector, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensors
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand, "In function nnl2_ad_gemmvp, multiplicand AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier, "In function nnl2_ad_gemmvp, multiplier AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(vector, "In function nnl2_ad_gemmvp, bias vector AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->data, "In function nnl2_ad_gemmvp, multiplicand tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->data, "In function nnl2_ad_gemmvp, multiplier tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(vector->data, "In function nnl2_ad_gemmvp, bias vector tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplicand->data->shape, "In function nnl2_ad_gemmvp, multiplicand tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(multiplier->data->shape, "In function nnl2_ad_gemmvp, multiplier tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(vector->data->shape, "In function nnl2_ad_gemmvp, bias vector tensor shape is NULL", NULL);
	#endif
	
	// Check that inputs have correct dimensions
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (multiplicand->data->rank != 2) {
			NNL2_ERROR("In function nnl2_ad_gemmvp, multiplicand must be a 2D matrix");
			return NULL;
		}
		
		if (multiplier->data->rank != 2) {
			NNL2_ERROR("In function nnl2_ad_gemmvp, multiplier must be a 2D matrix");
			return NULL;
		}
		
		if (vector->data->rank != 1) {
			NNL2_ERROR("In function nnl2_ad_gemmvp, bias vector must be a 1D vector");
			return NULL;
		}
	#endif
	
	// Extract dimensions
	int k_multiplicand = multiplicand->data->shape[1];  // columns of A
	int k_multiplier = multiplier->data->shape[0];      // rows of B
	int n = multiplier->data->shape[1];    // columns of B
	int vector_size = vector->data->shape[0]; // size of bias vector
	
	// Check dimension compatibility
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (k_multiplicand != k_multiplier) {
			NNL2_ERROR("In function nnl2_ad_gemmvp, inner dimensions don't match: %d != %d", k_multiplicand, k_multiplier);
			return NULL;
		}
		
		if (vector_size != n) {
			NNL2_ERROR("In function nnl2_ad_gemmvp, bias vector size %d doesn't match output columns %d", vector_size, n);
			return NULL;
		}
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute matrix multiplication with bias using GEMMVP
    result->data = nnl2_gemmvp(multiplicand->data, multiplier->data, vector->data);
    
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_gemmvp, failed to compute matrix multiplication with bias using nnl2_gemmvp");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_gemmvp, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
		nnl2_free_tensor(result->data); free(result);
		return NULL;
	}
	
	// Build computational graph if tracking is enabled
	if(track_graph) {
		result->num_roots = 3;
		result->roots = (nnl2_ad_tensor**)malloc(3 * sizeof(*result->roots));
		if(!result->roots) {
			NNL2_MALLOC_ERROR();
			nnl2_free_tensor(result->data);
			nnl2_free_tensor(result->grad);
			free(result);
			return NULL;
		}
	
	    // Set all input tensors as roots
		result->roots[0] = multiplicand;  // matrix A
		result->roots[1] = multiplier;    // matrix B
		result->roots[2] = vector;        // bias vector V
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_gemmvp;  break;
			
			default: {
				NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
				nnl2_free_ad_tensor(result);
				return NULL;
			}
		}
	} else {
		// No computational graph tracking
		result->num_roots = 0;
		result->roots = NULL;
		result->backward_fn = NULL;
	}
	
	// Initialize tensor metadata
    result->requires_grad = multiplicand->requires_grad || multiplier->requires_grad || vector->requires_grad;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
	
	result -> extra_field = NULL;
	result -> extra_free = NULL;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return result;
}

#endif /** NNL2_AD_GEMMVP_H **/
