#ifndef NNL2_AD_MSE_BACKWARD_H
#define NNL2_AD_MSE_BACKWARD_H

/** @file nnl2_ad_mse_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for MSE loss operation
 **/

/** @brief 
 * Compute derivative for MSE loss operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from MSE operation
 *
 ** @param prediction_tensor  
 * The prediction tensor to MSE operation
 *
 ** @param target_tensor
 * The target tensor to MSE operation
 *
 ** @details
 * dL/dprediction = (dL/doutput) * 2 * (prediction - target) / n_elements
 * dL/dtarget = (dL/doutput) * -2 * (prediction - target) / n_elements
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static void nnl2_ad_reverse_derivative_mse(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* prediction_tensor, nnl2_ad_tensor* target_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_mse, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor, "In function nnl2_ad_reverse_derivative_mse, prediction_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor, "In function nnl2_ad_reverse_derivative_mse, target_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_mse, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data, "In function nnl2_ad_reverse_derivative_mse, prediction_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data, "In function nnl2_ad_reverse_derivative_mse, target_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_mse, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->grad, "In function nnl2_ad_reverse_derivative_mse, prediction_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->grad, "In function nnl2_ad_reverse_derivative_mse, target_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mse, output_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(prediction_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mse, prediction_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(target_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mse, target_tensor shape is NULL");
    #endif
    
    // Calculate total number of elements
    size_t numel = product(prediction_tensor->data->shape, prediction_tensor->data->rank);
    
    if(numel == 0) {
        return;
    }
    
    // Output tensor should be scalar (MSE loss)
    if(output_tensor->data->shape[0] != 1) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
            NNL2_DEBUG("Output tensor for MSE should be scalar");
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
            nnl2_float64 scale = 2.0 * output_grad / (nnl2_float64)numel;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 diff = prediction_data[i] - target_data[i];
                
                if(prediction_tensor->requires_grad) {
                    prediction_grad_data[i] += scale * diff;
                }
                
                if(target_tensor->requires_grad) {
                    target_grad_data[i] += -scale * diff;
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
            nnl2_float32 scale = 2.0f * output_grad / (nnl2_float32)numel;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 diff = prediction_data[i] - target_data[i];
                
                if(prediction_tensor->requires_grad) {
                    prediction_grad_data[i] += scale * diff;
                }
                
                if(target_tensor->requires_grad) {
                    target_grad_data[i] += -scale * diff;
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

#endif /** NNL2_AD_MSE_BACKWARD_H **/
