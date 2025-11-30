#ifndef NNL2_AD_MSE_H
#define NNL2_AD_MSE_H

/** @file nnl2_ad_mse.h
 ** @brief AD implementation for MSE loss operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for MSE loss operation (stub version)
 *
 ** @param tensor
 * The output tensor from MSE operation that needs gradient computation
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 **/
static void nnl2_ad_reverse_backward_mse(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_mse, passed AD tensor is NULL");
    #endif
    
    // Call the actual derivative computation function
    nnl2_ad_reverse_derivative_mse(tensor, tensor->roots[0], tensor->roots[1]);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Create an automatic differentiation tensor for MSE loss operation
 *
 ** @param prediction 
 * Prediction tensor for MSE calculation
 *
 ** @param target 
 * Target tensor for MSE calculation (must match prediction shape and dtype)
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
 * If force=false: pointer to nnl2_ad_tensor containing MSE result
 * If force=true: pointer to scalar value of appropriate type
 * NULL on failure
 **/
void* nnl2_ad_mse(nnl2_ad_tensor* prediction, nnl2_ad_tensor* target, bool force, nnl2_ad_mode ad_mode, bool track_graph) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Basic null checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(prediction, "In function nnl2_ad_mse, prediction AD tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(target, "In function nnl2_ad_mse, target AD tensor is NULL", NULL);
    #endif
    
    // INT32 not supported
    if (prediction->data->dtype == INT32) {
        NNL2_ERROR("In function nnl2_ad_mse, INT32 dtype is not supported for MSE loss");
        return NULL;
    }
    
    // Compute MSE
    if (force) {
        // return scalar value, cannot continue graph
        size_t scalar_size = (prediction->data->dtype == FLOAT64) ? sizeof(nnl2_float64) : sizeof(nnl2_float32);
        void* scalar_result = malloc(scalar_size);
        if (!scalar_result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        // Compute MSE directly into allocated scalar memory
        nnl2_mse(prediction->data, target->data, scalar_result);
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return scalar_result;
        
    } else {
        // returns AD tensor that can continue computation graph
        nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
        if (!result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        result->magic_number = TENSOR_MAGIC_ALIVE;
        
        // result is a scalar tensor with shape [1]
        int32_t scalar_shape[] = {1};
        
        // Use the same dtype as input tensors
        nnl2_tensor_type result_dtype = prediction->data->dtype;
        result->data = nnl2_empty(scalar_shape, 1, result_dtype);
        if (!result->data) {
            NNL2_ERROR("In function nnl2_ad_mse, failed to allocate result tensor");
            free(result);
            return NULL;
        }
        
        // Compute MSE into the result tensor data
        nnl2_mse(prediction->data, target->data, result->data->data);
    
        // Allocate gradient tensor with same dtype
        result->grad = nnl2_empty(result->data->shape, result->data->rank, result_dtype);
        if (!result->grad) {
            NNL2_ERROR("In function nnl2_ad_mse, failed to allocate gradient tensor");
            nnl2_free_tensor(result->data);
            free(result);
            return NULL;
        }
    
        // Build computational graph if tracking is enabled
        if (track_graph) {
            result->num_roots = 2;
            result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(*result->roots));
            if (!result->roots) {
                NNL2_MALLOC_ERROR();
                nnl2_free_tensor(result->data);
                nnl2_free_tensor(result->grad);
                free(result);
                return NULL;
            }
    
            // Set input tensors as roots
            result->roots[0] = prediction;
            result->roots[1] = target;
        
            // Set the appropriate backward function based on AD mode
            switch(ad_mode) {
                case nnl2_ad_reverse_mode: 
                    result->backward_fn = nnl2_ad_reverse_backward_mse;  
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
        result->requires_grad = prediction->requires_grad || target->requires_grad;
        result->grad_initialized = false;
        result->is_leaf = false;
        
        // Common initialization
        result->name = NULL;
        result->ts_type = nnl2_type_ad;
        result->visited_gen = 0;
        result->extra_multiplier = 1.0f;
        result->extra_bool = false;
        result->extra_correspondence = NULL;
        result->extra_field = NULL;
        result->extra_free = NULL;
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return (void*)result;
    }
}

#endif /** NNL2_AD_MSE_H **/
