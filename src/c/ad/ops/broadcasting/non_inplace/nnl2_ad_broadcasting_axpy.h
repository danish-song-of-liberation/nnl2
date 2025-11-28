#ifndef NNL2_AD_AXPY_BROADCASTING_H
#define NNL2_AD_AXPY_BROADCASTING_H

/** @file nnl2_ad_axpy_broadcasting.h
 ** @brief AD implementation for AXPY with broadcasting operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for AXPY with broadcasting operation
 *
 ** @param tensor
 * The output tensor from axpy_broadcasting operation that needs gradient computation
 *
 ** @details
 * AXPY with broadcasting operation: result = alpha * x + y
 * Derivatives with broadcasting:
 * - d(result)/d(x) = alpha (with gradient summation for broadcasted dimensions)
 * - d(result)/d(y) = 1 (with gradient summation for broadcasted dimensions)
 * Propagates scaled gradients to both input tensors with proper broadcasting handling
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
 ** @see nnl2_ad_axpy_broadcasting()
 ** @see nnl2_ad_reverse_derivative_axpy_broadcasting()
 **/	
static void nnl2_ad_reverse_backward_axpy_broadcasting(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_axpy_broadcasting, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_axpy_broadcasting, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_axpy_broadcasting, axpyend root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[1], "In function nnl2_ad_reverse_backward_axpy_broadcasting, sumend root tensor is NULL");
	#endif
	
	// Compute the derivative of axpy_broadcasting operation and propagate to root tensors
    nnl2_ad_reverse_derivative_axpy_broadcasting(tensor, tensor->roots[0], tensor->roots[1], tensor->extra_multiplier);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for AXPY with broadcasting operation
 *
 ** @param axpyend 
 * First input tensor (x in alpha * x + y)
 *
 ** @param sumend 
 * Second input tensor (y in alpha * x + y)
 *
 ** @param multiplier 
 * Scaling factor (alpha in alpha * x + y)
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing result of alpha * x + y with broadcasting, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if axpyend or sumend is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor shapes are incompatible for broadcasting AXPY operation
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if axpy_broadcasting operation fails on input data
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
 ** @see axpy_broadcasting()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_axpy_broadcasting(nnl2_ad_tensor* axpyend, nnl2_ad_tensor* sumend, float multiplier, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensors
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(axpyend, "In function nnl2_ad_axpy_broadcasting, axpyend AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "In function nnl2_ad_axpy_broadcasting, sumend AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(axpyend->data, "In function nnl2_ad_axpy_broadcasting, axpyend tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "In function nnl2_ad_axpy_broadcasting, sumend tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(axpyend->data->shape, "In function nnl2_ad_axpy_broadcasting, axpyend tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data->shape, "In function nnl2_ad_axpy_broadcasting, sumend tensor shape is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute AXPY with broadcasting: alpha * x + y
    result->data = axpy_broadcasting(axpyend->data, sumend->data, multiplier);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_axpy_broadcasting, failed to compute AXPY with broadcasting");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_axpy_broadcasting, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
		nnl2_free_tensor(result->data); free(result);
		return NULL;
	}
	
	// Build computational graph if tracking is enabled
	if(track_graph) {
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
		result->roots[0] = axpyend;  // x in alpha * x + y
		result->roots[1] = sumend;   // y in alpha * x + y
		
		// Store multiplier for backward pass
		result->extra_multiplier = multiplier;
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_axpy_broadcasting;  break;
			
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
		result->extra_multiplier = 0.0f;
	}
	
	// Initialize tensor metadata
    result->requires_grad = axpyend->requires_grad || sumend->requires_grad;
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

#endif /** NNL2_AD_AXPY_BROADCASTING_H **/
