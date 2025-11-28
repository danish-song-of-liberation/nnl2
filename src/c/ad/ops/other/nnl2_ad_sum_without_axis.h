#ifndef NNL2_AD_SUM_WITHOUT_AXIS_H
#define NNL2_AD_SUM_WITHOUT_AXIS_H

/** @file nnl2_ad_sum_without_axis.h
 ** @brief AD implementation for sum without axis operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for sum without axis operation
 *
 ** @param tensor
 * The output tensor from sum operation that needs gradient computation
 *
 ** @details
 * Derivative of sum: d(sum)/dx = 1 for all elements
 * Propagates gradients to the root tensor using chain rule
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
 ** @see nnl2_ad_sum_without_axis()
 ** @see nnl2_ad_reverse_derivative_sum()
 **/	
static void nnl2_ad_reverse_backward_sum_without_axis(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_sum_without_axis, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_sum_without_axis, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_sum_without_axis, root tensor is NULL");
	#endif
	
	// Compute the derivative of sum operation and propagate to root tensor
    nnl2_ad_reverse_derivative_sum_without_axis(tensor, tensor->roots[0]);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for sum without axis operation or return scalar
 *
 ** @param input 
 * Input tensor for sum calculation
 *
 ** @param force 
 * If false: returns AD tensor with shape [1] that can continue computation graph
 * If true: returns scalar value (direct pointer to number) but cannot be differentiated
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return void*
 * - If force=false: pointer to nnl2_ad_tensor containing sum result
 * - If force=true: pointer to scalar value of appropriate type
 * - NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if input is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if sum_without_axis operation fails on input data
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
 ** @see nnl2_sum_without_axis()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
void* nnl2_ad_sum_without_axis(nnl2_ad_tensor* input, bool force, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensor
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input, "In function nnl2_ad_sum_without_axis, input AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input->data, "In function nnl2_ad_sum_without_axis, input AD tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input->data->shape, "In function nnl2_ad_sum_without_axis, input AD tensor shape is NULL", NULL);
	#endif
    
    // Compute sum
    if (force) {
        // Force mode: return scalar value, cannot continue graph
        void* scalar_result = malloc(sizeof(nnl2_float32));
        if(!scalar_result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        // Compute sum directly into allocated scalar memory
        nnl2_sum_without_axis(input->data, scalar_result);
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return scalar_result;
        
    } else {
        // Normal mode: returns AD tensor that can continue computation graph
        nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
        if(!result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        // To protect against re-freeing
        result->magic_number = TENSOR_MAGIC_ALIVE;
        
        // For sum, result is a scalar tensor with shape [1]
        int32_t scalar_shape[] = {1};
        result->data = nnl2_empty(scalar_shape, 1, input->data->dtype);
        if(!result->data) {
            NNL2_ERROR("In function nnl2_ad_sum_without_axis, failed to allocate result tensor");
            free(result);
            return NULL;
        }
        
        // Compute sum into the result tensor data
        // nnl2_sum_without_axis returns void, writes directly to result->data->data
        nnl2_sum_without_axis(input->data, result->data->data);
    
        // Allocate gradient tensor with same shape as result
        result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
        if(!result->grad) {
            NNL2_ERROR("In function nnl2_ad_sum_without_axis, failed to allocate gradient tensor");
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
            result->roots[0] = input;
        
            // Set the appropriate backward function based on AD mode
            switch(ad_mode) {
                case nnl2_ad_reverse_mode: 
                    result->backward_fn = nnl2_ad_reverse_backward_sum_without_axis;  
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
        }
    
        // Initialize tensor metadata
        result->requires_grad = input->requires_grad;
        result->grad_initialized = false;
        result->is_leaf = false;
        
        // Common initialization
        result->name = NULL;
        result->ts_type = nnl2_type_ad;
        result->visited_gen = 0;
        result->extra_multiplier = 1.0f;
        result->extra_bool = false;
        result->extra_correspondence = NULL;
		
		result -> extra_field = NULL;
		result -> extra_free = NULL;
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return (void*)result;
    }
}

#endif /** NNL2_AD_SUM_WITHOUT_AXIS_H **/
