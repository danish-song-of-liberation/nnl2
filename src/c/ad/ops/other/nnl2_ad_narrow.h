#ifndef NNL2_AD_NARROW_H
#define NNL2_AD_NARROW_H

/** @file nnl2_ad_narrow.h
 ** @brief AD implementation for narrow operation
 ** @date 2025
 ** @copyright MIT
 **/

// Structure to store narrow operation parameters
typedef struct {
    uint8_t dim;
    int32_t start;
    int32_t len;
} nnl2_narrow_args;

/** @brief 
 * Free function for narrow arguments
 *
 ** @param extra_object
 * Pointer to narrow arguments structure
 **/
static void nnl2_ad_free_narrow_args(void* extra_object) {
    if(extra_object) {
        free(extra_object);
    }
}

/** @brief 
 * Reverse mode backward pass for narrow operation
 *
 ** @param tensor
 * The output tensor from narrow operation that needs gradient computation
 *
 ** @details
 * Derivative of narrow: propagates gradients back to the corresponding slice
 * of the original tensor using chain rule
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
 * If tensor->extra_field is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_narrow()
 ** @see nnl2_ad_reverse_derivative_narrow()
 **/	
static void nnl2_ad_reverse_backward_narrow(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_narrow, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "In function nnl2_ad_reverse_backward_narrow, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots[0], "In function nnl2_ad_reverse_backward_narrow, root tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->extra_field, "In function nnl2_ad_reverse_backward_narrow, narrow arguments are NULL");
	#endif
	
	// Extract narrow parameters from extra_field
	nnl2_narrow_args* args = (nnl2_narrow_args*)tensor->extra_field;
	uint8_t dim = args->dim;
	int32_t start = args->start;
	int32_t len = args->len;
	
	// Compute the derivative of narrow operation and propagate to root tensor
    nnl2_ad_reverse_derivative_narrow(tensor, tensor->roots[0], dim, start, len);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for narrow operation
 *
 ** @param input 
 * Input tensor for narrow operation
 *
 ** @param dim 
 * Dimension along which to narrow
 *
 ** @param start 
 * Starting index for the narrow operation
 *
 ** @param len 
 * Length of the narrow slice
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *  
 ** @return nnl2_ad_tensor*
 * Pointer to nnl2_ad_tensor containing narrow result
 * NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if input is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if narrow operation fails on input data
 *
 ** @exception NNL2Error
 * Returns NULL if gradient tensor allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if roots array allocation fails when track_graph=true
 *
 ** @exception NNL2Error
 * Returns NULL if narrow arguments allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if unknown AD mode is specified
 *
 ** @see nnl2_narrow()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
nnl2_ad_tensor* nnl2_ad_narrow(nnl2_ad_tensor* input, uint8_t dim, int32_t start, int32_t len, nnl2_ad_mode ad_mode, bool track_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensor
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input, "In function nnl2_ad_narrow, input AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input->data, "In function nnl2_ad_narrow, input AD tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(input->data->shape, "In function nnl2_ad_narrow, input AD tensor shape is NULL", NULL);
	#endif
    
    // returns AD tensor that can continue computation graph
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        return NULL;
    }
    
    // To protect against re-freeing
    result->magic_number = TENSOR_MAGIC_ALIVE;
    
    // Perform narrow operation on input data
    result->data = nnl2_narrow(input->data, dim, start, len);
    if(!result->data) {
        NNL2_ERROR("In function nnl2_ad_narrow, failed to perform narrow operation");
        free(result);
        return NULL;
    }
    
    // Allocate gradient tensor with same shape as result
    result->grad = nnl2_empty(result->data->shape, result->data->rank, result->data->dtype);
    if(!result->grad) {
        NNL2_ERROR("In function nnl2_ad_narrow, failed to allocate gradient tensor");
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
                result->backward_fn = nnl2_ad_reverse_backward_narrow;  
                break;
            
            default: {
                NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
                nnl2_free_ad_tensor(result);
                return NULL;
            }
        }
        
        // Allocate and store narrow arguments in extra_field
        nnl2_narrow_args* args = malloc(sizeof(nnl2_narrow_args));
        if(!args) {
            NNL2_MALLOC_ERROR();
            nnl2_free_tensor(result->data);
            nnl2_free_tensor(result->grad);
            free(result->roots);
            free(result);
            return NULL;
        }
        
        args->dim = dim;
        args->start = start;
        args->len = len;
        
        result->extra_field = args;
        result->extra_free = nnl2_ad_free_narrow_args;
        
    } else {
        // No computational graph tracking
        result->num_roots = 0;
        result->roots = NULL;
        result->backward_fn = NULL;
        result->extra_field = NULL;
        result->extra_free = NULL;
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
    result->extra_integer = 0;
    result->extra_bool = false;
    result->extra_correspondence = NULL;
	
	result -> extra_field = NULL;
	result -> extra_free = NULL;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_AD_NARROW_H **/
