#ifndef NNL2_AD_SLICE_H
#define NNL2_AD_SLICE_H

/** @brief 
 * Reverse mode backward pass for slice operation
 *
 ** @param tensor
 * The output tensor from slice operation that needs gradient computation
 *
 ** @details
 * Propagates gradient back to input tensor by accumulating gradients in the sliced region.
 * Since slice creates a view, gradient accumulation is done efficiently.
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
 ** @see nnl2_ad_slice()
 ** @see nnl2_ad_reverse_derivative_slice()
 **/
static void nnl2_ad_reverse_backward_slice(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_slice, passed AD tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_slice, passed AD tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_slice, root tensor is NULL");
    #endif
    
    // Extract slice region from extra_correspondence
    if (tensor->extra_correspondence) {
        void** region_data = (void**)tensor->extra_correspondence;
        int32_t* from = (int32_t*)region_data[0];
        int32_t* to = (int32_t*)region_data[1];
        
        // Compute the derivative of slice operation and propagate to root tensor
        nnl2_ad_reverse_derivative_slice(tensor, tensor->roots[0], from, to);
    } else {
        NNL2_ERROR("In function nnl2_ad_reverse_backward_slice, extra_correspondence is NULL");
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Create an automatic differentiation tensor for slice operation (view version)
 *
 ** @param tensor 
 * Input AD tensor to slice
 *
 ** @param slice_from 
 * Starting indices for each dimension (array of rank elements)
 *
 ** @param slice_to 
 * Ending indices for each dimension (array of rank elements)
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * New AD tensor with sliced data view, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if input tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if slice_from or slice_to is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if nnl2_slice operation fails on input data
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
 ** @see nnl2_slice()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_slice(nnl2_ad_tensor* tensor, int32_t* slice_from, int32_t* slice_to, nnl2_ad_mode ad_mode, bool track_graph) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Basic null checks for input tensor and slice parameters
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_ad_slice, input AD tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(slice_from, "In function nnl2_ad_slice, slice_from array is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(slice_to, "In function nnl2_ad_slice, slice_to array is NULL", NULL);
    #endif
    
    // Comprehensive null checks for tensor components
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "In function nnl2_ad_slice, input AD tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data->shape, "In function nnl2_ad_slice, input AD tensor shape is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        return NULL;
    }
    
    // To protect against re-freeing
    result->magic_number = TENSOR_MAGIC_ALIVE;
    
    // Slice the tensor data (creates view, O(1) operation)
    result->data = nnl2_slice(tensor->data, slice_from, slice_to);
    if(!result->data) {
        NNL2_ERROR("In function nnl2_ad_slice, failed to slice tensor using nnl2_slice");
        free(result);
        return NULL;
    }
    
    // Allocate gradient tensor with same sliced shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
    if(!result->grad) {
        NNL2_ERROR("In function nnl2_ad_slice, failed to allocate nnl2 empty tensor for gradient using nnl2_empty function");
        nnl2_free_tensor(result->data);
        free(result);
        return NULL;
    }
    
    // Store slice region in extra_correspondence for backward pass
    int32_t rank = tensor->data->rank;
    void** region_data = malloc(2 * sizeof(void*));
    if(!region_data) {
        NNL2_MALLOC_ERROR();
        nnl2_free_tensor(result->data);
        nnl2_free_tensor(result->grad);
        free(result);
        return NULL;
    }
    
    // Copy from and to arrays
    int32_t* from_copy = malloc(rank * sizeof(int32_t));
    int32_t* to_copy = malloc(rank * sizeof(int32_t));
    if(!from_copy || !to_copy) {
        NNL2_MALLOC_ERROR();
        free(region_data);
        if(from_copy) free(from_copy);
        if(to_copy) free(to_copy);
        nnl2_free_tensor(result->data);
        nnl2_free_tensor(result->grad);
        free(result);
        return NULL;
    }
    
    memcpy(from_copy, slice_from, rank * sizeof(int32_t));
    memcpy(to_copy, slice_to, rank * sizeof(int32_t));
    
    region_data[0] = from_copy;
    region_data[1] = to_copy;
    result->extra_correspondence = region_data;
    
    // Build computational graph if tracking is enabled
    if(track_graph) {
        result->num_roots = 1;
        result->roots = (nnl2_ad_tensor**)malloc(sizeof(*result->roots));
        if(!result->roots) {
            NNL2_MALLOC_ERROR();
            free(from_copy);
            free(to_copy);
            free(region_data);
            nnl2_free_tensor(result->data);
            nnl2_free_tensor(result->grad);
            free(result);
            return NULL;
        }
    
        // Set input tensor as root
        result->roots[0] = tensor;
        
        // Set the appropriate backward function based on AD mode
        switch(ad_mode) {
            case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_slice; break;
            
            default: {
                NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
                free(from_copy);
                free(to_copy);
                free(region_data);
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
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_AD_SLICE_H **/
