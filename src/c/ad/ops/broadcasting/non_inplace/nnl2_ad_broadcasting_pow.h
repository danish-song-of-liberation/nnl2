#ifndef NNL2_AD_POW_BROADCASTING_H
#define NNL2_AD_POW_BROADCASTING_H

/** @file nnl2_ad_pow_broadcasting.h
 ** @brief AD implementation for power operation with broadcasting
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for power operation with broadcasting
 *
 ** @param tensor
 * The output tensor from pow_broadcasting operation that needs gradient computation
 *
 ** @details
 * Derivative of a^b with broadcasting follows generalized power rule:
 * da = b * a^(b-1) * dout (with broadcasting)
 * db = a^b * ln(a) * dout (with broadcasting) for a > 0
 * Broadcasting dimensions receive summed gradients (reduction)
 * Special handling required for edge cases (a = 0, b = 0, etc.)
 * Propagates gradients to both root tensors with proper broadcasting handling
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->roots[0] or tensor->roots[1] is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_pow_broadcasting()
 ** @see nnl2_ad_reverse_derivative_pow_broadcasting()
 **/	
static void nnl2_ad_reverse_backward_pow_broadcasting(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_pow_broadcasting, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_pow_broadcasting, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_pow_broadcasting, base root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[1], "In function nnl2_ad_reverse_backward_pow_broadcasting, exponent root tensor is NULL");
	#endif
	
	// Compute the derivative of pow_broadcasting operation and propagate to root tensors
    nnl2_ad_reverse_derivative_pow_broadcasting(tensor, tensor->roots[0], tensor->roots[1]);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for power operation with broadcasting
 *
 ** @param base 
 * First input tensor (base)
 *
 ** @param exponent 
 * Second input tensor (exponent)
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing base raised to exponent with broadcasting, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if base or exponent is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor shapes are incompatible for broadcasting power operation
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if pow_broadcasting operation fails on input data (e.g., negative base with non-integer exponent)
 *
 ** @exception NNL2Error
 * Returns NULL if gradient tensor allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if roots array allocation fails when track_graph=true and requires_grad=true
 *
 ** @exception NNL2Error
 * Returns NULL if unknown AD mode is specified
 *
 ** @warning
 * Special care must be taken with domain restrictions:
 * For non-integer exponents, base must be non-negative
 * For exponent = 0, result is 1 (including 0^0 which is typically defined as 1)
 * Gradient computation requires additional handling for edge cases
 *
 ** @note
 * The computational graph is built only when both track_graph=true and result->requires_grad=true
 * This optimization avoids unnecessary memory allocation when gradients are not needed
 *
 ** @see pow_broadcasting()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_pow_broadcasting(nnl2_ad_tensor* base, nnl2_ad_tensor* exponent, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensors
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base, "In function nnl2_ad_pow_broadcasting, base AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent, "In function nnl2_ad_pow_broadcasting, exponent AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base->data, "In function nnl2_ad_pow_broadcasting, base tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent->data, "In function nnl2_ad_pow_broadcasting, exponent tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(base->data->shape, "In function nnl2_ad_pow_broadcasting, base tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(exponent->data->shape, "In function nnl2_ad_pow_broadcasting, exponent tensor shape is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute power with broadcasting
    result->data = pow_broadcasting(base->data, exponent->data);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_pow_broadcasting, failed to compute power with broadcasting");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_pow_broadcasting, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
		nnl2_free_tensor(result->data); free(result);
		return NULL;
	}
	
	// Determine if gradients are required for this operation
	result->requires_grad = base->requires_grad || exponent->requires_grad;
	
	// Build computational graph if tracking is enabled AND gradients are required
	if(track_graph && result->requires_grad) {
		result->num_roots = 2;
		result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(*result->roots));
		if(!result->roots) {
			NNL2_MALLOC_ERROR();
			nnl2_free_tensor(result->data);
			nnl2_free_tensor(result->grad);
			free(result);
			return NULL;
		}
	
	    // Set both input tensors as roots
		result->roots[0] = base;
		result->roots[1] = exponent;
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: 
				result->backward_fn = nnl2_ad_reverse_backward_pow_broadcasting;  
				break;
			
			default: {
				NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
				nnl2_free_ad_tensor(result);
				return NULL;
			}
		}
	} else {
		// No computational graph tracking or gradients not required
		result->num_roots = 0;
		result->roots = NULL;
		result->backward_fn = NULL;
	}
	
	// Initialize remaining tensor metadata
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

#endif /** NNL2_AD_POW_BROADCASTING_H **/
