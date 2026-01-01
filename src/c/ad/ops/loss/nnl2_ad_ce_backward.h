#ifndef NNL2_AD_CROSS_ENTROPY_BACKWARD_H
#define NNL2_AD_CROSS_ENTROPY_BACKWARD_H

/** @file nnl2_ad_cross_entropy_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for Cross-Entropy loss operation
 **/

/** @brief 
 * Compute derivative for Cross-Entropy loss operation in reverse mode for class indices target
 *
 ** @param output_tensor
 * The output tensor from Cross-Entropy operation (scalar loss)
 *
 ** @param prediction_tensor  
 * The prediction tensor to Cross-Entropy operation (logits, shape: [batch, classes])
 *
 ** @param target_tensor
 * The target tensor to Cross-Entropy operation (class indices, shape: [batch])
 *
 ** @details
 * For indices target:
 * dL/dprediction_i = (dL/doutput) * (softmax(prediction_i) - delta(i == target)) / batch_size
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static void nnl2_ad_reverse_derivative_cross_entropy_indices(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* prediction_tensor, nnl2_ad_tensor* target_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, prediction_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, target_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, prediction_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, target_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, prediction_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_indices, target_tensor grad is NULL");
    #endif
    
    int32_t batch_size = prediction_tensor->data->shape[0];
    int32_t num_classes = prediction_tensor->data->shape[1];
    
    if(batch_size == 0 || num_classes == 0) {
        return;
    }
    
    // Output tensor should be scalar (Cross-Entropy loss)
    if(output_tensor->data->shape[0] != 1) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
            NNL2_DEBUG("Output tensor for Cross-Entropy should be scalar");
        #endif
        return;
    }
    
    switch(prediction_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64* output_grad_data = (nnl2_float64*)output_tensor->grad->data;
            nnl2_float64* prediction_data = (nnl2_float64*)prediction_tensor->data->data;
            nnl2_float64* prediction_grad_data = (nnl2_float64*)prediction_tensor->grad->data;
            nnl2_int32* target_data = (nnl2_int32*)target_tensor->data->data;
            
            nnl2_float64 output_grad = output_grad_data[0];
            nnl2_float64 scale = output_grad / (nnl2_float64)batch_size;
            
            for(int32_t b = 0; b < batch_size; b++) {
                nnl2_int32 true_class = target_data[b];
                
                // Compute softmax for current sample
                nnl2_float64 max_logit = prediction_data[b * num_classes];
                for(int32_t c = 1; c < num_classes; c++) {
                    if(prediction_data[b * num_classes + c] > max_logit) {
                        max_logit = prediction_data[b * num_classes + c];
                    }
                }
                
                nnl2_float64 sum_exp = 0.0;
                for(int32_t c = 0; c < num_classes; c++) {
                    sum_exp += exp(prediction_data[b * num_classes + c] - max_logit);
                }
                
                // Compute gradients: ∂L/∂prediction = softmax(prediction) - one_hot(true_class)
                for(int32_t c = 0; c < num_classes; c++) {
                    nnl2_float64 softmax_val = exp(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                    nnl2_float64 gradient = softmax_val - (c == true_class ? 1.0 : 0.0);
                    
                    if(prediction_tensor->requires_grad) {
                        prediction_grad_data[b * num_classes + c] += scale * gradient;
                    }
                }
            }
            
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* output_grad_data = (nnl2_float32*)output_tensor->grad->data;
            nnl2_float32* prediction_data = (nnl2_float32*)prediction_tensor->data->data;
            nnl2_float32* prediction_grad_data = (nnl2_float32*)prediction_tensor->grad->data;
            nnl2_int32* target_data = (nnl2_int32*)target_tensor->data->data;
            
            nnl2_float32 output_grad = output_grad_data[0];
            nnl2_float32 scale = output_grad / (nnl2_float32)batch_size;
            
            for(int32_t b = 0; b < batch_size; b++) {
                nnl2_int32 true_class = target_data[b];
                
                // Compute softmax for current sample
                nnl2_float32 max_logit = prediction_data[b * num_classes];
                for(int32_t c = 1; c < num_classes; c++) {
                    if(prediction_data[b * num_classes + c] > max_logit) {
                        max_logit = prediction_data[b * num_classes + c];
                    }
                }
                
                nnl2_float32 sum_exp = 0.0f;
                for(int32_t c = 0; c < num_classes; c++) {
                    sum_exp += expf(prediction_data[b * num_classes + c] - max_logit);
                }
                
                // Compute gradients: ∂L/∂prediction = softmax(prediction) - one_hot(true_class)
                for(int32_t c = 0; c < num_classes; c++) {
                    nnl2_float32 softmax_val = expf(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                    nnl2_float32 gradient = softmax_val - (c == true_class ? 1.0f : 0.0f);
                    
                    if(prediction_tensor->requires_grad) {
                        prediction_grad_data[b * num_classes + c] += scale * gradient;
                    }
                }
            }
            
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(prediction_tensor->data->dtype);
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Compute derivative for Cross-Entropy loss operation in reverse mode for one-hot target
 *
 ** @param output_tensor
 * The output tensor from Cross-Entropy operation (scalar loss)
 *
 ** @param prediction_tensor  
 * The prediction tensor to Cross-Entropy operation (logits, shape: [batch, classes])
 *
 ** @param target_tensor
 * The target tensor to Cross-Entropy operation (one-hot, shape: [batch, classes])
 *
 ** @details
 * For one-hot target:
 * dL/dprediction_i = (dL/doutput) * (softmax(prediction_i) - target_i) / batch_size
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static void nnl2_ad_reverse_derivative_cross_entropy_onehot(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* prediction_tensor, nnl2_ad_tensor* target_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, prediction_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, target_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, prediction_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, target_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, prediction_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->grad, "In function nnl2_ad_reverse_derivative_cross_entropy_onehot, target_tensor grad is NULL");
    #endif
    
    int32_t batch_size = prediction_tensor->data->shape[0];
    int32_t num_classes = prediction_tensor->data->shape[1];
    
    if(batch_size == 0 || num_classes == 0) {
        return;
    }
    
    // Output tensor should be scalar (Cross-Entropy loss)
    if(output_tensor->data->shape[0] != 1) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
            NNL2_DEBUG("Output tensor for Cross-Entropy should be scalar");
        #endif
        return;
    }
    
    switch(prediction_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64* output_grad_data = (nnl2_float64*)output_tensor->grad->data;
            nnl2_float64* prediction_data = (nnl2_float64*)prediction_tensor->data->data;
            nnl2_float64* prediction_grad_data = (nnl2_float64*)prediction_tensor->grad->data;
            
            nnl2_float64 output_grad = output_grad_data[0];
            nnl2_float64 scale = output_grad / (nnl2_float64)batch_size;
            
            for(int32_t b = 0; b < batch_size; b++) {
                // Compute softmax for current sample
                nnl2_float64 max_logit = prediction_data[b * num_classes];
                for(int32_t c = 1; c < num_classes; c++) {
                    if(prediction_data[b * num_classes + c] > max_logit) {
                        max_logit = prediction_data[b * num_classes + c];
                    }
                }
                
                nnl2_float64 sum_exp = 0.0;
                for(int32_t c = 0; c < num_classes; c++) {
                    sum_exp += exp(prediction_data[b * num_classes + c] - max_logit);
                }
                
                // Compute gradients based on target type
                if(target_tensor->data->dtype == FLOAT64) {
                    nnl2_float64* target_data = (nnl2_float64*)target_tensor->data->data;
                    
                    for(int32_t c = 0; c < num_classes; c++) {
                        nnl2_float64 softmax_val = exp(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                        nnl2_float64 gradient = softmax_val - target_data[b * num_classes + c];
                        
                        if(prediction_tensor->requires_grad) {
                            prediction_grad_data[b * num_classes + c] += scale * gradient;
                        }
                        
                        if(target_tensor->requires_grad) {
                            // Gradient for target (usually not needed, but computed for completeness)
                            nnl2_float64* target_grad_data = (nnl2_float64*)target_tensor->grad->data;
                            target_grad_data[b * num_classes + c] += -scale * prediction_data[b * num_classes + c];
                        }
                    }
                } else if(target_tensor->data->dtype == FLOAT32) {
                    nnl2_float32* target_data = (nnl2_float32*)target_tensor->data->data;
                    
                    for(int32_t c = 0; c < num_classes; c++) {
                        nnl2_float64 softmax_val = exp(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                        nnl2_float64 gradient = softmax_val - (nnl2_float64)target_data[b * num_classes + c];
                        
                        if(prediction_tensor->requires_grad) {
                            prediction_grad_data[b * num_classes + c] += scale * gradient;
                        }
                    }
                }
            }
            
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* output_grad_data = (nnl2_float32*)output_tensor->grad->data;
            nnl2_float32* prediction_data = (nnl2_float32*)prediction_tensor->data->data;
            nnl2_float32* prediction_grad_data = (nnl2_float32*)prediction_tensor->grad->data;
            
            nnl2_float32 output_grad = output_grad_data[0];
            nnl2_float32 scale = output_grad / (nnl2_float32)batch_size;
            
            for(int32_t b = 0; b < batch_size; b++) {
                // Compute softmax for current sample
                nnl2_float32 max_logit = prediction_data[b * num_classes];
                for(int32_t c = 1; c < num_classes; c++) {
                    if(prediction_data[b * num_classes + c] > max_logit) {
                        max_logit = prediction_data[b * num_classes + c];
                    }
                }
                
                nnl2_float32 sum_exp = 0.0f;
                for(int32_t c = 0; c < num_classes; c++) {
                    sum_exp += expf(prediction_data[b * num_classes + c] - max_logit);
                }
                
                // Compute gradients based on target type
                if(target_tensor->data->dtype == FLOAT32) {
                    nnl2_float32* target_data = (nnl2_float32*)target_tensor->data->data;
                    
                    for(int32_t c = 0; c < num_classes; c++) {
                        nnl2_float32 softmax_val = expf(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                        nnl2_float32 gradient = softmax_val - target_data[b * num_classes + c];
                        
                        if(prediction_tensor->requires_grad) {
                            prediction_grad_data[b * num_classes + c] += scale * gradient;
                        }
                        
                        if(target_tensor->requires_grad) {
                            // Gradient for target (usually not needed, but computed for completeness)
                            nnl2_float32* target_grad_data = (nnl2_float32*)target_tensor->grad->data;
                            target_grad_data[b * num_classes + c] += -scale * prediction_data[b * num_classes + c];
                        }
                    }
                } else if(target_tensor->data->dtype == FLOAT64) {
                    nnl2_float64* target_data = (nnl2_float64*)target_tensor->data->data;
                    
                    for(int32_t c = 0; c < num_classes; c++) {
                        nnl2_float32 softmax_val = expf(prediction_data[b * num_classes + c] - max_logit) / sum_exp;
                        nnl2_float32 gradient = softmax_val - (nnl2_float32)target_data[b * num_classes + c];
                        
                        if(prediction_tensor->requires_grad) {
                            prediction_grad_data[b * num_classes + c] += scale * gradient;
                        }
                    }
                }
            }
            
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(prediction_tensor->data->dtype);
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Main dispatcher for Cross-Entropy derivative computation
 *
 ** @param output_tensor
 * The output tensor from Cross-Entropy operation
 *
 ** @param prediction_tensor  
 * The prediction tensor to Cross-Entropy operation
 *
 ** @param target_tensor
 * The target tensor to Cross-Entropy operation
 *
 ** @details
 * Dispatches to appropriate derivative function based on target format
 *
 ** @see nnl2_ad_reverse_derivative_cross_entropy_indices
 ** @see nnl2_ad_reverse_derivative_cross_entropy_onehot
 **/
void nnl2_ad_reverse_derivative_cross_entropy(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* prediction_tensor, nnl2_ad_tensor* target_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Check target format from extra_field
    if (output_tensor->extra_field) {
        int target_format = *((int*)output_tensor->extra_field);
        
        if (target_format == 0) {  // Indices format
            nnl2_ad_reverse_derivative_cross_entropy_indices(output_tensor, prediction_tensor, target_tensor);
        } else {  // One-hot format
            nnl2_ad_reverse_derivative_cross_entropy_onehot(output_tensor, prediction_tensor, target_tensor);
        }
    } else {
        // Default to indices format for backward compatibility
        nnl2_ad_reverse_derivative_cross_entropy_indices(output_tensor, prediction_tensor, target_tensor);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_CROSS_ENTROPY_BACKWARD_H **/
