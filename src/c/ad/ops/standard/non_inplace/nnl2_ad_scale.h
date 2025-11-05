#ifndef NNL2_AD_SCALE_H
#define NNL2_AD_SCALE_H

/** @file nnl2_ad_scale.h
 ** @brief AD implementation for scaling operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for scaling operation
 *
 ** @param tensor
 * The output tensor from scale operation that needs gradient computation
 *
 ** @details
 * Derivative of k*x is: k (where k is the multiplier)
 * Propagates scaled gradient to root tensor using chain rule
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->roots[0] is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_scale()
 ** @see nnl2_ad_reverse_derivative_scale()
 **/	
static void nnl2_ad_reverse_backward_scale(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_scale, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_scale, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_scale, root tensor is NULL");
	#endif
	
	// Compute the derivative of scale operation and propagate to root tensor
    nnl2_ad_reverse_derivative_scale(tensor, tensor->roots[0], tensor->extra_multiplier);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for scaling operation
 *
 ** @param ad_tensor 
 * Input tensor
 *
 ** @param multiplier
 * Scaling factor to multiply the tensor by
 *
 ** @param save_type 
 * Whether to preserve input data type in output
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing scaled values, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if ad_tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if ad_tensor->shape is NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if ad_tensor->data is NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if scale operation fails on input data
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
 ** @see scale()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_scale(nnl2_ad_tensor* ad_tensor, nnl2_float32 multiplier, bool save_type, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null check for input tensor
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_scale, passed AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_scale, passed AD tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_scale, passed AD tensor data is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute scaled values
    result->data = scale(ad_tensor->data, multiplier, save_type);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_scale, failed to apply scaling to the tensor data");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape and type as input
    result->grad = nnl2_empty(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_scale, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
		nnl2_free_tensor(result->data); free(result);
		return NULL;
	}
	
	// Build computational graph if tracking is enabled
	if(track_graph) {
		result->num_roots = 1;
		result->roots = (nnl2_ad_tensor**)malloc(sizeof(*result->roots));
		if(!result->roots) {
			NNL2_MALLOC_ERROR();
			nnl2_free_tensor(result->data);
			nnl2_free_tensor(result->grad);
			free(result);
			return NULL;
		}
	
	    // Set the input tensor as root
		result->roots[0] = ad_tensor; 
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_scale;  break;
			
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
    result->requires_grad = ad_tensor->requires_grad;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    result->extra_multiplier = multiplier;  // Store multiplier for backward pass
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return result;
}

#endif /** NNL2_AD_SCALE_H **/
