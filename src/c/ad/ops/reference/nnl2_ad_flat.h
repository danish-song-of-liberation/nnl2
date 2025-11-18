#ifndef NNL2_AD_FLAT_H
#define NNL2_AD_FLAT_H

/** @file nnl2_ad_flat.h
 ** @brief AD implementation for flat tensor element access operation
 ** @date 2025
 ** @copyright MIT License
 **/

/** @brief 
 * Reverse mode backward pass for flat operation
 *
 ** @param tensor
 * The output tensor from flat operation that needs gradient computation
 *
 ** @details
 * Derivative of flat: propagates gradient back to original tensor using stored flat index
 * Handles gradient accumulation for flat tensor element access operations
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
 ** @see nnl2_ad_flat()
 ** @see nnl2_ad_reverse_derivative_flat()
 **/
static void nnl2_ad_reverse_backward_flat(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_flat, passed AD tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor -> data, "In function nnl2_ad_reverse_backward_flat, passed AD tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor -> roots[0], "In function nnl2_ad_reverse_backward_flat, root tensor is NULL");
    #endif
    
    size_t* flat_index = (size_t*)tensor -> extra_correspondence;
    
    // Compute the derivative of flat operation
    nnl2_ad_reverse_derivative_flat(tensor, tensor -> roots[0], *flat_index);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Create an automatic differentiation tensor for flat tensor element access operation
 *
 ** @param tensor 
 * Input tensor for flat operation
 *
 ** @param at 
 * Flat index for element access
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *
 ** @param force
 * If true, returns raw scalar value instead of AD tensor
 *  
 ** @return void*
 * New AD tensor containing flat result as scalar tensor, or scalar value if force=true, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor components are NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if flat operation fails on input tensor
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
 ** @see nnl2_get_raw_tensor_elem_at()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
void* nnl2_ad_flat(nnl2_ad_tensor* tensor, size_t at, nnl2_ad_mode ad_mode, bool track_graph, bool force) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Basic null checks for input tensors
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_ad_flat, AD tensor is NULL", NULL);
    #endif
    
    // Comprehensive null checks for tensor components
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor -> data, "In function nnl2_ad_flat, AD tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor -> data -> shape, "In function nnl2_ad_flat, AD tensor shape is NULL", NULL);
    #endif
    
    // Always returns a scalar element, so force returns the raw pointer
    if(force) {
        void* scalar_result = nnl2_get_raw_tensor_elem_at(tensor->data, at);
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return scalar_result;
        
    } else {
        nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
        if(!result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        // To protect against re-freeing
        result -> magic_number = TENSOR_MAGIC_ALIVE;
        
        // Create a scalar tensor with shape [1] containing the element value
        result -> data = nnl2_empty((int32_t[]){1}, 1, tensor -> data -> dtype);
        if(!result->data) {
            NNL2_ERROR("In function nnl2_ad_flat, failed to allocate result tensor");
            free(result);
            return NULL;
        }
        
        // Get the element value and store it in the scalar tensor
        void* elem = nnl2_get_raw_tensor_elem_at(tensor->data, at);
        if(!elem) {
            NNL2_ERROR("In function nnl2_ad_flat, failed to get tensor element");
            nnl2_free_tensor(result->data);
            free(result);
            return NULL;
        }
        
        // Copy the element value to our scalar tensor
        size_t dtype_size = get_dtype_size(tensor->data->dtype);
        memcpy(result->data->data, elem, dtype_size);
        
        // Allocate gradient tensor with same shape as result (scalar [1])
        result -> grad = nnl2_empty(result -> data -> shape, result -> data -> rank, result -> data -> dtype);
        if(!result -> grad) {
            NNL2_ERROR("In function nnl2_ad_flat, failed to allocate gradient tensor");
            nnl2_free_tensor(result->data);
            free(result);
            return NULL;
        }
        
        // Build computational graph if tracking is enabled
        if(track_graph) {
            result -> num_roots = 1;
            result -> roots = (nnl2_ad_tensor**)malloc(sizeof(*result->roots));
            if(!result -> roots) {
                NNL2_MALLOC_ERROR();
                nnl2_free_tensor(result -> data);
                nnl2_free_tensor(result -> grad);
                free(result);
                return NULL;
            }
        
            result -> roots[0] = tensor;
            
            // Set the appropriate backward function based on AD mode
            switch(ad_mode) {
                case nnl2_ad_reverse_mode: {
                    result -> backward_fn = nnl2_ad_reverse_backward_flat;  
                    break;
                }
                
                default: {
                    NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
                    nnl2_free_ad_tensor(result);
                    return NULL;
                }
            }
        } else {
            // No computational graph tracking
            result -> num_roots = 0;
            result -> roots = NULL;
            result -> backward_fn = NULL;
        }
        
        // Store flat index in extra_correspondence (need to allocate since it's size_t)
        size_t* stored_index = (size_t*)malloc(sizeof(size_t));
        if(!stored_index) {
            NNL2_MALLOC_ERROR();
            nnl2_free_ad_tensor(result);
            return NULL;
        }
		
        *stored_index = at;
        
        // Initialize tensor metadata
        result -> requires_grad = tensor -> requires_grad;
        result -> grad_initialized = false;
        result -> is_leaf = false;
        result -> extra_multiplier = 1.0f;
        result -> extra_correspondence = (void*)stored_index;
        result -> name = NULL;
        result -> ts_type = nnl2_type_ad;
        result -> visited_gen = 0;
        result -> extra_integer = 0; // Not used for flat
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return result;
    }
}

#endif /** NNL2_AD_FLAT_H **/
