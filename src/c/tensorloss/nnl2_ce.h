#ifndef NNL2_CROSS_ENTROPY_H
#define NNL2_CROSS_ENTROPY_H

// NNL2

/** @brief 
 * Computes Cross-Entropy Loss for class indices target format
 *
 ** @param prediction 
 * Pointer to prediction tensor (logits, shape: [batch_size, num_classes])
 *
 ** @param target 
 * Pointer to target tensor (class indices, shape: [batch_size])
 *
 ** @param record 
 * Pointer to memory where result will be stored
 *
 ** @note
 * Assumes prediction is FLOAT32 or FLOAT64 and target is INT32
 */
static void nnl2_naive_cross_entropy_indices(nnl2_tensor* prediction, nnl2_tensor* target, void* record) {
    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];
    nnl2_int32* target_indices = (nnl2_int32*)target->data;
    
    switch(prediction->dtype) {
        case FLOAT32: {
            nnl2_float32 total_loss = 0.0f;
            nnl2_float32* pred_data = (nnl2_float32*)prediction->data;
            
            for(size_t i = 0; i < batch_size; i++) {
                nnl2_int32 true_class = target_indices[i];
                
                // Find max 
                nnl2_float32 max_logit = pred_data[i * num_classes];
                for(size_t j = 1; j < num_classes; j++) {
                    nnl2_float32 logit = pred_data[i * num_classes + j];
                    if (logit > max_logit) max_logit = logit;
                }
                
                // log-sum-exp
                nnl2_float32 sum_exp = 0.0f;
                for(size_t j = 0; j < num_classes; j++) {
                    sum_exp += expf(pred_data[i * num_classes + j] - max_logit);
                }
                
                nnl2_float32 log_sum_exp = logf(sum_exp) + max_logit;
                nnl2_float32 true_logit = pred_data[i * num_classes + true_class];
                
                total_loss += log_sum_exp - true_logit;
            }
            
            nnl2_float32* result = (nnl2_float32*)record;
            *result = (batch_size > 0) ? total_loss / (nnl2_float32)batch_size : 0.0f;
            break;
        }
        
        case FLOAT64: {
            nnl2_float64 total_loss = 0.0;
            nnl2_float64* pred_data = (nnl2_float64*)prediction->data;
            
            for(size_t i = 0; i < batch_size; i++) {
                nnl2_int32 true_class = target_indices[i];
                
                // Find max 
                nnl2_float64 max_logit = pred_data[i * num_classes];
                for(size_t j = 1; j < num_classes; j++) {
                    nnl2_float64 logit = pred_data[i * num_classes + j];
                    if (logit > max_logit) max_logit = logit;
                }
                
                // log-sum-exp
                nnl2_float64 sum_exp = 0.0;
                for(size_t j = 0; j < num_classes; j++) {
                    sum_exp += exp(pred_data[i * num_classes + j] - max_logit);
                }
                
                nnl2_float64 log_sum_exp = log(sum_exp) + max_logit;
                nnl2_float64 true_logit = pred_data[i * num_classes + true_class];
                
                total_loss += log_sum_exp - true_logit;
            }
            
            nnl2_float64* result = (nnl2_float64*)record;
            *result = (batch_size > 0) ? total_loss / (nnl2_float64)batch_size : 0.0;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(prediction->dtype);
            break;
        }
    }
}

/** @brief 
 * Computes Cross-Entropy Loss for one-hot encoded target format
 *
 ** @param prediction 
 * Pointer to prediction tensor (logits, shape: [batch_size, num_classes])
 *
 ** @param target 
 * Pointer to target tensor (one-hot encoded, shape: [batch_size, num_classes])
 *
 ** @param record 
 * Pointer to memory where result will be stored
 *
 ** @note
 * Supports mixed precision between prediction and target
 */
static void nnl2_naive_cross_entropy_onehot(nnl2_tensor* prediction, nnl2_tensor* target, void* record) {
    size_t batch_size = prediction->shape[0];
    size_t num_classes = prediction->shape[1];
    
    switch(prediction->dtype) {
        case FLOAT32: {
            nnl2_float32 total_loss = 0.0f;
            nnl2_float32* pred_data = (nnl2_float32*)prediction->data;
            
            if (target->dtype == FLOAT32) {
                nnl2_float32* target_data = (nnl2_float32*)target->data;
                
                for(size_t i = 0; i < batch_size; i++) {
                    // Find max
                    nnl2_float32 max_logit = pred_data[i * num_classes];
                    for(size_t j = 1; j < num_classes; j++) {
                        nnl2_float32 logit = pred_data[i * num_classes + j];
                        if (logit > max_logit) max_logit = logit;
                    }
                    
                    // log-sum-exp
                    nnl2_float32 sum_exp = 0.0f;
                    for(size_t j = 0; j < num_classes; j++) {
                        sum_exp += expf(pred_data[i * num_classes + j] - max_logit);
                    }
                    
                    nnl2_float32 log_sum_exp = logf(sum_exp) + max_logit;
                    
                    // Sum over classes where target is non-zero
                    nnl2_float32 loss = 0.0f;
                    for(size_t j = 0; j < num_classes; j++) {
                        nnl2_float32 target_val = target_data[i * num_classes + j];
                        if (target_val > 0.0f) {
                            loss += target_val * (log_sum_exp - pred_data[i * num_classes + j]);
                        }
                    }
                    
                    total_loss += loss;
                }
            }
			
            else if (target->dtype == FLOAT64) {
                nnl2_float64* target_data = (nnl2_float64*)target->data;
                
                for(size_t i = 0; i < batch_size; i++) {
                    // Find max
                    nnl2_float32 max_logit = pred_data[i * num_classes];
                    for(size_t j = 1; j < num_classes; j++) {
                        nnl2_float32 logit = pred_data[i * num_classes + j];
                        if (logit > max_logit) max_logit = logit;
                    }
                    
                    // log-sum-exp
                    nnl2_float32 sum_exp = 0.0f;
                    for(size_t j = 0; j < num_classes; j++) {
                        sum_exp += expf(pred_data[i * num_classes + j] - max_logit);
                    }
                    
                    nnl2_float32 log_sum_exp = logf(sum_exp) + max_logit;
                    
                    nnl2_float32 loss = 0.0f;
                    for(size_t j = 0; j < num_classes; j++) {
                        nnl2_float64 target_val = target_data[i * num_classes + j];
                        if (target_val > 0.0) {
                            loss += (nnl2_float32)target_val * (log_sum_exp - pred_data[i * num_classes + j]);
                        }
                    }
                    
                    total_loss += loss;
                }
            }
            else {
                NNL2_ERROR("Unsupported target dtype for one-hot cross-entropy with FLOAT32 prediction");
                return;
            }
            
            nnl2_float32* result = (nnl2_float32*)record;
            *result = (batch_size > 0) ? total_loss / (nnl2_float32)batch_size : 0.0f;
            break;
        }
        
        case FLOAT64: {
            nnl2_float64 total_loss = 0.0;
            nnl2_float64* pred_data = (nnl2_float64*)prediction->data;
            
            if(target->dtype == FLOAT64) {
                nnl2_float64* target_data = (nnl2_float64*)target->data;
                
                for(size_t i = 0; i < batch_size; i++) {
                    // Find max 
                    nnl2_float64 max_logit = pred_data[i * num_classes];
                    for(size_t j = 1; j < num_classes; j++) {
                        nnl2_float64 logit = pred_data[i * num_classes + j];
                        if (logit > max_logit) max_logit = logit;
                    }
                    
                    // log-sum-exp
                    nnl2_float64 sum_exp = 0.0;
                    for(size_t j = 0; j < num_classes; j++) {
                        sum_exp += exp(pred_data[i * num_classes + j] - max_logit);
                    }
                    
                    nnl2_float64 log_sum_exp = log(sum_exp) + max_logit;
                    
                    nnl2_float64 loss = 0.0;
                    for(size_t j = 0; j < num_classes; j++) {
                        nnl2_float64 target_val = target_data[i * num_classes + j];
                        if (target_val > 0.0) {
                            loss += target_val * (log_sum_exp - pred_data[i * num_classes + j]);
                        }
                    }
                    
                    total_loss += loss;
                }
				
            } else if(target->dtype == FLOAT32) {
                nnl2_float32* target_data = (nnl2_float32*)target->data;
                
                for(size_t i = 0; i < batch_size; i++) {
                    // Find max 
                    nnl2_float64 max_logit = pred_data[i * num_classes];
                    for(size_t j = 1; j < num_classes; j++) {
                        nnl2_float64 logit = pred_data[i * num_classes + j];
                        if (logit > max_logit) max_logit = logit;
                    }
                    
                    nnl2_float64 sum_exp = 0.0;
                    for(size_t j = 0; j < num_classes; j++) {
                        sum_exp += exp(pred_data[i * num_classes + j] - max_logit);
                    }
                    
                    nnl2_float64 log_sum_exp = log(sum_exp) + max_logit;
                    
                    nnl2_float64 loss = 0.0;
                    for(size_t j = 0; j < num_classes; j++) {
                        nnl2_float32 target_val = target_data[i * num_classes + j];
                        if (target_val > 0.0f) {
                            loss += (nnl2_float64)target_val * (log_sum_exp - pred_data[i * num_classes + j]);
                        }
                    }
                    
                    total_loss += loss;
                }
				
            } else {
                NNL2_ERROR("Unsupported target dtype for one-hot cross-entropy with FLOAT64 prediction");
                return;
            }
            
            nnl2_float64* result = (nnl2_float64*)record;
            *result = (batch_size > 0) ? total_loss / (nnl2_float64)batch_size : 0.0;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(prediction->dtype);
            break;
        }
    }
}

/** @brief 
 * Dispatcher for Cross-Entropy Loss between prediction and target tensors
 *
 ** @param prediction 
 * Pointer to prediction tensor (logits, shape: [batch_size, num_classes])
 *
 ** @param target 
 * Pointer to target tensor (class indices or one-hot encoded)
 *
 ** @param record 
 * Pointer to memory where result will be stored
 */
void nnl2_naive_cross_entropy(nnl2_tensor* prediction, nnl2_tensor* target, void* record) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
	    NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(prediction->rank < 2) {
			NNL2_ERROR("Prediction tensor must have rank >= 2 for cross-entropy");
			return;
		}
		
		NNL2_CHECK_NULL_IF_ERR_RETURN(prediction, "In function nnl2_naive_cross_entropy, passed prediction is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(target, "In function nnl2_naive_cross_entropy, passed target is NULL");
	#endif 
    
    int32_t batch_size = prediction->shape[0];
    int32_t num_classes = prediction->shape[1];
    
    int target_is_indices =(target->rank == 1 && target->shape[0] == batch_size);
    int target_is_onehot = (target->rank == 2 && target->shape[0] == batch_size && target->shape[1] == num_classes);
    
    if(!target_is_indices && !target_is_onehot) {
        NNL2_ERROR("Target tensor must be either class indices [batch_size] or one-hot [batch_size, num_classes]");
        return;
    }
    
    if(target_is_indices) {
        if(target->dtype != INT32) {
            NNL2_ERROR("Class indices target must be INT32 dtype");
            return;
        }
        
        nnl2_int32* indices = (nnl2_int32*)target->data;
        for(int32_t i = 0; i < batch_size; i++) {
            if(indices[i] < 0 || indices[i] >= (nnl2_int32)num_classes) {
                NNL2_ERROR("Class index out of bounds at position %zu: %d (num_classes: %zu)", i, indices[i], num_classes);
                return;
            }
        }
        
        nnl2_naive_cross_entropy_indices(prediction, target, record);
		
    } else { // target_is_onehot
        if(target->dtype != FLOAT32 && target->dtype != FLOAT64) {
            NNL2_ERROR("One-hot target must be FLOAT32 or FLOAT64 dtype");
            return;
        }
        
        nnl2_naive_cross_entropy_onehot(prediction, target, record);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
	    NNL2_FUNC_EXIT();
	#endif 
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for Cross-Entropy loss operation
 * @details
 * Array follows the standard backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_cross_entropy: Dispatcher implementation
 *
 * @see nnl2_naive_cross_entropy
 * @see nnl2_naive_cross_entropy_indices
 * @see nnl2_naive_cross_entropy_onehot
 */
Implementation cross_entropy_backends[] = {
    REGISTER_BACKEND(nnl2_naive_cross_entropy, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for Cross-Entropy loss operation
 * @ingroup backend_system
 */
cross_entropyfn nnl2_cross_entropy;

/**
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(cross_entropy);

/**
 * @brief Sets the backend for Cross-Entropy loss operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_cross_entropy_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(cross_entropy_backends, nnl2_cross_entropy, backend_name, CURRENT_BACKEND(cross_entropy));
}

/**
 * @brief Gets the name of the active backend for Cross-Entropy loss operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_cross_entropy_backend() {
    return CURRENT_BACKEND(cross_entropy);
}

/**
 * @brief Function declaration for getting all `cross_entropy` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(cross_entropy);

/**
 * @brief Function declaration for getting the number of all `cross_entropy` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(cross_entropy);

#endif /** NNL2_CROSS_ENTROPY_H **/
