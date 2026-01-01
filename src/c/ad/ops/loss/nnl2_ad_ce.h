#ifndef NNL2_AD_CROSS_ENTROPY_H
#define NNL2_AD_CROSS_ENTROPY_H

/** @file nnl2_ad_cross_entropy.h
 ** @brief AD implementation for Cross-Entropy loss operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Reverse mode backward pass for Cross-Entropy loss operation (stub version)
 *
 ** @param tensor
 * The output tensor from Cross-Entropy operation that needs gradient computation
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 */
static void nnl2_ad_reverse_backward_cross_entropy(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_cross_entropy, passed AD tensor is NULL");
    #endif
    
    // Call the actual derivative computation function
    nnl2_ad_reverse_derivative_cross_entropy(tensor, tensor->roots[0], tensor->roots[1]);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Create an automatic differentiation tensor for Cross-Entropy loss operation
 *
 ** @param prediction 
 * Prediction tensor for Cross-Entropy calculation (logits, shape: [batch_size, num_classes])
 *
 ** @param target 
 * Target tensor for Cross-Entropy calculation (class indices [batch_size] or one-hot [batch_size, num_classes])
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
 * If force=false: pointer to nnl2_ad_tensor containing Cross-Entropy result
 * If force=true: pointer to scalar value of appropriate type
 * NULL on failure
 */
void* nnl2_ad_cross_entropy(nnl2_ad_tensor* prediction, nnl2_ad_tensor* target, bool force, nnl2_ad_mode ad_mode, bool track_graph) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Basic null checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(prediction, "In function nnl2_ad_cross_entropy, prediction AD tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(target, "In function nnl2_ad_cross_entropy, target AD tensor is NULL", NULL);
    #endif
    
    // Check tensor shapes
    if (prediction->data->rank < 2) {
        NNL2_ERROR("In function nnl2_ad_cross_entropy, prediction tensor must have rank >= 2");
        return NULL;
    }
    
    int32_t batch_size = prediction->data->shape[0];
    int32_t num_classes = prediction->data->shape[1];
    
    int target_is_indices = (target->data->rank == 1 && target->data->shape[0] == batch_size);
    int target_is_onehot = (target->data->rank == 2 && target->data->shape[0] == batch_size && target->data->shape[1] == num_classes);
    
    if (!target_is_indices && !target_is_onehot) {
        NNL2_ERROR("In function nnl2_ad_cross_entropy, target tensor must be either class indices [batch_size] or one-hot [batch_size, num_classes]");
        return NULL;
    }
    
    // INT32 not supported for prediction (only FLOAT32/FLOAT64)
    if (prediction->data->dtype == INT32) {
        NNL2_ERROR("In function nnl2_ad_cross_entropy, INT32 dtype is not supported for prediction tensor");
        return NULL;
    }
    
    // Validate target types
    if (target_is_indices && target->data->dtype != INT32) {
        NNL2_ERROR("In function nnl2_ad_cross_entropy, class indices target must be INT32 dtype");
        return NULL;
    }
    
    if (target_is_onehot && target->data->dtype != FLOAT32 && target->data->dtype != FLOAT64) {
        NNL2_ERROR("In function nnl2_ad_cross_entropy, one-hot target must be FLOAT32 or FLOAT64 dtype");
        return NULL;
    }
    
    // Compute Cross-Entropy
    if (force) {
        // return scalar value, cannot continue graph
        size_t scalar_size = (prediction->data->dtype == FLOAT64) ? sizeof(nnl2_float64) : sizeof(nnl2_float32);
        void* scalar_result = malloc(scalar_size);
        if (!scalar_result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
        
        // Compute Cross-Entropy directly into allocated scalar memory
        nnl2_cross_entropy(prediction->data, target->data, scalar_result);
        
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
        
        // Use the same dtype as prediction tensor
        nnl2_tensor_type result_dtype = prediction->data->dtype;
        result->data = nnl2_empty(scalar_shape, 1, result_dtype);
        if (!result->data) {
            NNL2_ERROR("In function nnl2_ad_cross_entropy, failed to allocate result tensor");
            free(result);
            return NULL;
        }
        
        // Compute Cross-Entropy into the result tensor data
        nnl2_cross_entropy(prediction->data, target->data, result->data->data);
    
        // Allocate gradient tensor with same dtype
        result->grad = nnl2_empty(result->data->shape, result->data->rank, result_dtype);
        if (!result->grad) {
            NNL2_ERROR("In function nnl2_ad_cross_entropy, failed to allocate gradient tensor");
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
            
            // Store target format information for backward pass
            int* target_format = malloc(sizeof(int));
            if (!target_format) {
                NNL2_MALLOC_ERROR();
                nnl2_free_tensor(result->data);
                nnl2_free_tensor(result->grad);
                free(result->roots);
                free(result);
                return NULL;
            }
            
            *target_format = target_is_indices ? 0 : 1; // 0 = indices, 1 = one-hot
            result->extra_field = target_format;
            result->extra_free = free;
        
            // Set the appropriate backward function based on AD mode
            switch(ad_mode) {
                case nnl2_ad_reverse_mode: 
                    result->backward_fn = nnl2_ad_reverse_backward_cross_entropy;  
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
            result->extra_field = NULL;
            result->extra_free = NULL;
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
        if (!track_graph) {
            result->extra_field = NULL;
            result->extra_free = NULL;
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return (void*)result;
    }
}

#endif /** NNL2_AD_CROSS_ENTROPY_H **/
