#ifndef NNL2_AD_ADD_BROADCASTING_H
#define NNL2_AD_ADD_BROADCASTING_H

/** @file nnl2_ad_add_broadcasting.h
 ** @brief AD implementation for addition with broadcasting operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for addition with broadcasting operation
 *
 ** @param tensor
 * The output tensor from add_broadcasting operation that needs gradient computation
 *
 ** @details
 * Derivative of a + b with broadcasting is: 1 for both operands
 * Broadcasting dimensions receive summed gradients (reduction)
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
 ** @see nnl2_ad_add_broadcasting()
 ** @see nnl2_ad_reverse_derivative_add_broadcasting()
 **/	
static void nnl2_ad_reverse_backward_add_broadcasting(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_add_broadcasting, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_add_broadcasting, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_add_broadcasting, addend root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[1], "In function nnl2_ad_reverse_backward_add_broadcasting, sumend root tensor is NULL");
	#endif
	
	// Compute the derivative of add_broadcasting operation and propagate to root tensors
    nnl2_ad_reverse_derivative_add_broadcasting(tensor, tensor->roots[0], tensor->roots[1]);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for addition with broadcasting operation
 *
 ** @param addend 
 * First input tensor
 *
 ** @param sumend 
 * Second input tensor
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing sum of inputs with broadcasting, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if addend or sumend is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor shapes are incompatible for broadcasting addition
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if add_broadcasting operation fails on input data
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
 ** @see add_broadcasting()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_add_broadcasting(nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensors
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(addend, "In function nnl2_ad_add_broadcasting, addend AD tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend, "In function nnl2_ad_add_broadcasting, sumend AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(addend->data, "In function nnl2_ad_add_broadcasting, addend tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data, "In function nnl2_ad_add_broadcasting, sumend tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(addend->data->shape, "In function nnl2_ad_add_broadcasting, addend tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(sumend->data->shape, "In function nnl2_ad_add_broadcasting, sumend tensor shape is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute addition with broadcasting
    result->data = add_broadcasting(addend->data, sumend->data);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_add_broadcasting, failed to compute addition with broadcasting");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_add_broadcasting, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
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
		result->roots[0] = addend;
		result->roots[1] = sumend;
		
		// Set the appropriate backward function based on AD mode
		switch(ad_mode) {
			case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_add_broadcasting;  break;
			
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
    result->requires_grad = addend->requires_grad || sumend->requires_grad;
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

#endif /** NNL2_AD_ADD_BROADCASTING_H **/