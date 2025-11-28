#ifndef NNL2_AD_RESHAPE_H
#define NNL2_AD_RESHAPE_H

/** @file nnl2_ad_reshape.h
 ** @brief AD implementation for reshape operation (copy version)
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for reshape operation
 *
 ** @param tensor
 * The output tensor from reshape operation that needs gradient computation
 *
 ** @details
 * Propagates gradient back to input tensor by reshaping gradient to match input shape
 * Since reshape creates a copy, gradient propagation requires reshaping back
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
 ** @see nnl2_ad_reshape()
 ** @see nnl2_ad_reverse_derivative_reshape()
 **/
static void nnl2_ad_reverse_backward_reshape(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_reshape, passed AD tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_reshape, passed AD tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_reshape, root tensor is NULL");
    #endif
    
    // Compute the derivative of reshape operation and propagate to root tensor
    nnl2_ad_reverse_derivative_reshape(tensor, tensor->roots[0]);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Create an automatic differentiation tensor for reshape operation (copy version)
 *
 ** @param tensor 
 * Input AD tensor to reshape
 *
 ** @param new_shape 
 * New shape array for the tensor
 *
 ** @param new_shape_len 
 * Length of the new_shape array
 *
 ** @param force 
 * Whether to force reshape even if total elements don't match
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor with reshaped data, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if input tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if new_shape is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if nnl2_reshape operation fails on input data
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
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
 ** @see nnl2_reshape()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_reshape(nnl2_ad_tensor* tensor, int32_t* new_shape, int32_t new_shape_len, bool force, nnl2_ad_mode ad_mode, bool track_graph) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Basic null checks for input tensor and shape
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_ad_reshape, input AD tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(new_shape, "In function nnl2_ad_reshape, new_shape array is NULL", NULL);
    #endif
    
    // Comprehensive null checks for tensor components
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "In function nnl2_ad_reshape, input AD tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data->shape, "In function nnl2_ad_reshape, input AD tensor shape is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        return NULL;
    }
    
    // To protect against re-freeing
    result->magic_number = TENSOR_MAGIC_ALIVE;
    
    // Reshape the tensor data (creates copy)
    result->data = nnl2_reshape(tensor->data, new_shape, new_shape_len, force);
    if(!result->data) {
        NNL2_ERROR("In function nnl2_ad_reshape, failed to reshape tensor using nnl2_reshape");
        free(result);
        return NULL;
    }
    
    // Allocate gradient tensor with same new shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
    if(!result->grad) {
        NNL2_ERROR("In function nnl2_ad_reshape, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
        nnl2_free_tensor(result->data);
        free(result);
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
    
        // Set input tensor as root
        result->roots[0] = tensor;
        
        // Set the appropriate backward function based on AD mode
        switch(ad_mode) {
            case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_reshape; break;
            
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
    result->requires_grad = tensor->requires_grad;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    result->visited_gen = 0;
    result->extra_multiplier = 0.0f;
    result->extra_bool = false;
    result->extra_correspondence = NULL;
	
	result -> extra_field = NULL;
	result -> extra_free = NULL;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_AD_RESHAPE_H **/
