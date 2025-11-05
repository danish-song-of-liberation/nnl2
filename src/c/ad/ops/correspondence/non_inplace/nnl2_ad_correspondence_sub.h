#ifndef NNL2_AD_SUB_CORRESPONDENCE_H
#define NNL2_AD_SUB_CORRESPONDENCE_H

/** @file nnl2_ad_sub_correspondence.h
 ** @brief AD implementation for subtraction with correspondence operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for subtraction with correspondence operation
 *
 ** @param tensor
 * The output tensor from sub_correspondence operation that needs gradient computation
 *
 ** @details
 * Subtraction with correspondence operation: result = tensor - dec
 * Derivative: 1 (gradient passes through unchanged)
 * Propagates gradient to root tensor using chain rule
 * The subtrahend (value to subtract) is stored in extra_correspondence field
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
 ** @exception NNL2Error
 * If tensor->extra_correspondence is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_sub_correspondence()
 ** @see nnl2_ad_reverse_derivative_sub_correspondence()
 **/	
static void nnl2_ad_reverse_backward_sub_correspondence(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_sub_correspondence, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_sub_correspondence, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_sub_correspondence, root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->extra_correspondence, "In function nnl2_ad_reverse_backward_sub_correspondence, extra_correspondence is NULL");
	#endif
	
	// Compute the derivative of sub_correspondence operation and propagate to root tensor
    nnl2_ad_reverse_derivative_sub_correspondence(tensor, tensor->roots[0], tensor->extra_correspondence);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for subtraction with correspondence operation
 *
 ** @param tensor 
 * Input tensor (minuend in tensor - dec)
 *
 ** @param dec 
 * Pointer to subtrahend data (correspondence structure - value to subtract)
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing result of tensor - dec, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if dec is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor->shape is NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor->data is NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if sub_decf operation fails on input data
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
 ** @see sub_decf()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_sub_correspondence(nnl2_ad_tensor* tensor, void* dec, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input parameters
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_ad_sub_correspondence, passed AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dec, "In function nnl2_ad_sub_correspondence, dec (subtrahend) is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "In function nnl2_ad_sub_correspondence, passed AD tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data->shape, "In function nnl2_ad_sub_correspondence, passed AD tensor shape is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// Initialize all fields to default values first
	result->magic_number = TENSOR_MAGIC_ALIVE;
	result->data = NULL;
	result->grad = NULL;
	result->num_roots = 0;
	result->roots = NULL;
	result->backward_fn = NULL;
	result->extra_correspondence = NULL;
	result->requires_grad = false;
	result->grad_initialized = false;
	result->is_leaf = false;
	result->name = NULL;
	result->ts_type = nnl2_type_ad;
    
	// Compute subtraction with correspondence: tensor - dec
    result->data = sub_decf(tensor->data, dec);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_sub_correspondence, failed to apply subtraction with correspondence to the tensor data");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape and type as input
    result->grad = nnl2_empty(tensor->data->shape, tensor->data->rank, tensor->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_sub_correspondence, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
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
		result->roots[0] = tensor; 
		
		// Store the subtrahend for backward pass
		result->extra_correspondence = dec;
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: 
				result->backward_fn = nnl2_ad_reverse_backward_sub_correspondence;  
				break;
			
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
		result->extra_correspondence = NULL;
	}
	
	// Initialize tensor metadata
    result->requires_grad = tensor->requires_grad;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
    
    return result;
}

#endif /** NNL2_AD_SUB_CORRESPONDENCE_H **/
