#ifndef NNL2_AD_LEAKYRELU_H
#define NNL2_AD_LEAKYRELU_H

// NNL2

/** @file nnl2_ad_leakyrelu.h
 ** @brief AD implementation for LeakyReLU operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for LeakyReLU operation
 *
 ** @param tensor
 * The output tensor from LeakyReLU operation that needs gradient computation
 *
 ** @details
 * Derivative of LeakyReLU(x) is: 1 if x >= 0, otherwise alpha if x < 0
 * Uses extra_multiplier field to store alpha parameter for backward pass
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data->shape is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->roots[0] is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_leakyrelu()
 ** @see nnl2_ad_reverse_derivative_leakyrelu()
 **/	
static void nnl2_ad_reverse_backward_leakyrelu(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_leakyrelu, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_leakyrelu, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data->shape, "In function nnl2_ad_reverse_backward_leakyrelu, passed AD tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_leakyrelu, passed AD tensor root is NULL");
	#endif
	
	// Compute the derivative of LeakyReLU operation and propagate to root tensor
    nnl2_ad_reverse_derivative_leakyrelu(tensor, tensor->roots[0], tensor->extra_multiplier);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}   

/** @brief 
 * Create an automatic differentiation tensor for LeakyReLU operation
 *
 ** @param ad_tensor 
 * Input tensor
 *
 ** @param alpha
 * Negative slope coefficient for x < 0
 *
 ** @param save_type
 * Whether to preserve input data type in output
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor containing LeakyReLU values, or NULL on failure
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
 * Returns NULL if leakyrelu operation fails on input data
 *
 ** @exception NNL2Error
 * Returns NULL if gradient tensor allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if roots array allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if unknown AD mode is specified
 *
 ** @see leakyrelu()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_leakyrelu(nnl2_ad_tensor* ad_tensor, float alpha, bool save_type, nnl2_ad_mode ad_mode) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null check for input tensor
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_leakyrelu, passed AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_leakyrelu, passed AD tensor shape is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_leakyrelu, passed AD tensor data is NULL", NULL);
	#endif
	
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	if(!result) {
		NNL2_MALLOC_ERROR();
		return NULL;
	}
	
	// To protect against re-freeing
	result->magic_number = TENSOR_MAGIC_ALIVE;
    
	// Compute LeakyReLU values
    result->data = leakyrelu(ad_tensor->data, alpha, save_type);
	if(!result->data) {
		NNL2_ERROR("In function nnl2_ad_leakyrelu, failed to apply leakyrelu to the tensor data");
		free(result);
		return NULL;
	}
	
	// Allocate gradient tensor with same shape and type as input
    result->grad = nnl2_empty(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype);
	if(!result->grad) {
		NNL2_ERROR("In function nnl2_ad_leakyrelu, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
		nnl2_free_tensor(result->data); free(result);
		return NULL;
	}
	
	// Build computational graph
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
	
	// Store alpha parameter for backward pass
	result->extra_multiplier = alpha;
	
	// Set the appropriate backward function based on AD mode
	switch(ad_mode) {
		case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_leakyrelu; break;
		
		default: {
			NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
			nnl2_free_ad_tensor(result);
			return NULL;
		}
	}
	
	// Initialize tensor metadata
    result->requires_grad = ad_tensor->requires_grad;
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

#endif /** NNL2_AD_LEAKYRELU_H **/

