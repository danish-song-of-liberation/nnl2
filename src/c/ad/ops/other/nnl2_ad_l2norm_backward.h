#ifndef NNL2_AD_L2NORM_BACKWARD_H
#define NNL2_AD_L2NORM_BACKWARD_H

/** @file nnl2_ad_l2norm_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for L2 norm operation
 **/

/** @brief 
 * Compute derivative for L2 norm operation in reverse mode
 *
 ** @param output_tensor
 * The output tensor from l2norm operation
 *
 ** @param input_tensor  
 * The input tensor to l2norm operation
 *
 ** @details
 * Computes: dL/dinput = (dL/doutput) * (input / norm(input))
 * Where norm(input) is the L2 norm computed earlier
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @exception NNL2Error
 * If any tensor is NULL, function returns early
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_l2norm(nnl2_ad_tensor* output_tensor, nnl2_ad_tensor* input_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!input_tensor->requires_grad) {
        return;
    }
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor, "In function nnl2_ad_reverse_derivative_l2norm, output_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor, "In function nnl2_ad_reverse_derivative_l2norm, input_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data, "In function nnl2_ad_reverse_derivative_l2norm, output_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data, "In function nnl2_ad_reverse_derivative_l2norm, input_tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->grad, "In function nnl2_ad_reverse_derivative_l2norm, output_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->grad, "In function nnl2_ad_reverse_derivative_l2norm, input_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(output_tensor->data->shape, "In function nnl2_ad_reverse_derivative_l2norm, output_tensor shape is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(input_tensor->data->shape, "In function nnl2_ad_reverse_derivative_l2norm, input_tensor shape is NULL");
    #endif
    
    // Calculate total number of elements in input tensor
    size_t numel = nnl2_product(input_tensor->data->shape, input_tensor->data->rank);
    
    // Output tensor should be scalar (L2 norm)
    if(output_tensor->data->shape[0] != 1) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
            NNL2_DEBUG("Output tensor for L2 norm should be scalar");
        #endif
        return;
    }
    
    switch(input_tensor->data->dtype) {
        case FLOAT64: {
            // Type-cast
            nnl2_float64* output_grad_data = (nnl2_float64*)output_tensor->grad->data;
            nnl2_float64* output_data = (nnl2_float64*)output_tensor->data->data;
            nnl2_float64* input_data = (nnl2_float64*)input_tensor->data->data;
            nnl2_float64* input_grad_data = (nnl2_float64*)input_tensor->grad->data;
            
            nnl2_float64 norm_value = output_data[0];
            
            // Avoid division by zero
            if(norm_value == 0.0) {
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
                    NNL2_DEBUG("L2 norm is zero, skipping gradient computation to avoid division by zero");
                #endif
                return;
            }
            
            nnl2_float64 output_grad = output_grad_data[0];
            
            for(size_t i = 0; i < numel; i++) {
                // dL/dinput = (dL/doutput) * (input / norm)
                input_grad_data[i] += output_grad * (input_data[i] / norm_value);
            }
            
            break;
        }
        
        case FLOAT32: {
            // Type-cast
            nnl2_float32* output_grad_data = (nnl2_float32*)output_tensor->grad->data;
            nnl2_float32* output_data = (nnl2_float32*)output_tensor->data->data;
            nnl2_float32* input_data = (nnl2_float32*)input_tensor->data->data;
            nnl2_float32* input_grad_data = (nnl2_float32*)input_tensor->grad->data;
            
            nnl2_float32 norm_value = output_data[0];
            
            // Avoid division by zero
            if(norm_value == 0.0f) {
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
                    NNL2_DEBUG("L2 norm is zero, skipping gradient computation to avoid division by zero");
                #endif
                return;
            }
            
            nnl2_float32 output_grad = output_grad_data[0];
            
            for(size_t i = 0; i < numel; i++) {
                // dL/dinput = (dL/doutput) * (input / norm)
                input_grad_data[i] += output_grad * (input_data[i] / norm_value);
            }
            
            break;
        }
        
        case INT32: {
            // Type-cast
            nnl2_int32* output_grad_data = (nnl2_int32*)output_tensor->grad->data;
            nnl2_int32* output_data = (nnl2_int32*)output_tensor->data->data;
            nnl2_int32* input_data = (nnl2_int32*)input_tensor->data->data;
            nnl2_int32* input_grad_data = (nnl2_int32*)input_tensor->grad->data;
            
            nnl2_int32 norm_value = output_data[0];
            
            // Avoid division by zero
            if(norm_value == 0) {
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
                    NNL2_DEBUG("L2 norm is zero, skipping gradient computation to avoid division by zero");
                #endif
                return;
            }
            
            nnl2_int32 output_grad = output_grad_data[0];
            
            for(size_t i = 0; i < numel; i++) {
                // dL/dinput = (dL/doutput) * (input / norm)
                input_grad_data[i] += output_grad * (input_data[i] / norm_value);
            }
            
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(input_tensor->data->dtype);
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_L2NORM_BACKWARD_H **/
