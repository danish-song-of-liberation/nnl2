#ifndef NNL2_AD_MAE_BACKWARD_H
#define NNL2_AD_MAE_BACKWARD_H

/** @file nnl2_ad_mae_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for MAE loss operation
 **/

/** @brief 
 * Compute derivative for MAE loss operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from MAE operation
 *
 ** @param prediction_tensor  
 * The prediction tensor to MAE operation
 *
 ** @param target_tensor
 * The target tensor to MAE operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static void nnl2_ad_reverse_derivative_mae(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* prediction_tensor, nnl2_ad_tensor* target_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_mae, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor, "In function nnl2_ad_reverse_derivative_mae, prediction_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor, "In function nnl2_ad_reverse_derivative_mae, target_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_mae, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data, "In function nnl2_ad_reverse_derivative_mae, prediction_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data, "In function nnl2_ad_reverse_derivative_mae, target_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_mae, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->grad, "In function nnl2_ad_reverse_derivative_mae, prediction_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->grad, "In function nnl2_ad_reverse_derivative_mae, target_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mae, output_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mae, prediction_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mae, target_tensor shape is NULL");
    #endif
    
    // Calculate total number of elements
    size_t numel = product(prediction_tensor->data->shape, prediction_tensor->data->rank);
    
    if(numel == 0) {
        return;
    }
    
    // Output tensor should be scalar (MAE loss)
    if(output_tensor->data->shape[0] != 1) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
            NNL2_DEBUG("Output tensor for MAE should be scalar");
        #endif
        return;
    }
    
    switch(prediction_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64* output_grad_data = (nnl2_float64*)output_tensor->grad->data;
            nnl2_float64* prediction_data = (nnl2_float64*)prediction_tensor->data->data;
            nnl2_float64* target_data = (nnl2_float64*)target_tensor->data->data;
            nnl2_float64* prediction_grad_data = (nnl2_float64*)prediction_tensor->grad->data;
            nnl2_float64* target_grad_data = (nnl2_float64*)target_tensor->grad->data;
            
            nnl2_float64 output_grad = output_grad_data[0];
            nnl2_float64 scale = output_grad / (nnl2_float64)numel;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 diff = prediction_data[i] - target_data[i];
                nnl2_float64 sign;
                
                // 1 if x > 0, -1 if x < 0, 0 if x == 0
                if(diff > 0.0) {
                    sign = 1.0;
                } else if(diff < 0.0) {
                    sign = -1.0;
                } else {
                    sign = 0.0;
                }
                
                if(prediction_tensor->requires_grad) {
                    prediction_grad_data[i] += scale * sign;
                }
                
                if(target_tensor->requires_grad) {
                    target_grad_data[i] += -scale * sign;
                }
            }
            
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* output_grad_data = (nnl2_float32*)output_tensor->grad->data;
            nnl2_float32* prediction_data = (nnl2_float32*)prediction_tensor->data->data;
            nnl2_float32* target_data = (nnl2_float32*)target_tensor->data->data;
            nnl2_float32* prediction_grad_data = (nnl2_float32*)prediction_tensor->grad->data;
            nnl2_float32* target_grad_data = (nnl2_float32*)target_tensor->grad->data;
            
            nnl2_float32 output_grad = output_grad_data[0];
            nnl2_float32 scale = output_grad / (nnl2_float32)numel;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 diff = prediction_data[i] - target_data[i];
                nnl2_float32 sign;
                
                // 1 if x > 0, -1 if x < 0, 0 if x == 0
                if(diff > 0.0f) {
                    sign = 1.0f;
                } else if(diff < 0.0f) {
                    sign = -1.0f;
                } else {
                    sign = 0.0f;
                }
                
                if(prediction_tensor->requires_grad) {
                    prediction_grad_data[i] += scale * sign;
                }
                
                if(target_tensor->requires_grad) {
                    target_grad_data[i] += -scale * sign;
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

#endif /** NNL2_AD_MAE_BACKWARD_H **/
