#ifndef NNL2_AD_DIV_BACKWARD_H
#define NNL2_AD_DIV_BACKWARD_H

/** @file nnl2_ad_div_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for division operation
 **/

/** @brief 
 * Computes the gradient of the division operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the division operation
 *
 ** @param dividend 
 * The dividend input tensor to the division operation
 *
 ** @param divisor 
 * The divisor input tensor to the division operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_div(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* dividend, nnl2_ad_tensor* divisor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_div, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "In function nnl2_ad_reverse_derivative_div, dividend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "In function nnl2_ad_reverse_derivative_div, divisor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_div, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "In function nnl2_ad_reverse_derivative_div, dividend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "In function nnl2_ad_reverse_derivative_div, divisor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_div, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data->shape, "In function nnl2_ad_reverse_derivative_div, dividend shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data->shape, "In function nnl2_ad_reverse_derivative_div, divisor shape is NULL");
	#endif
    
    // Calculate total number of elements in tensors
    size_t numel = product(out_tensor->data->shape, out_tensor->data->rank);
    nnl2_tensor_type dtype = out_tensor->data->dtype;

    switch(dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* out_grad_data = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* dividend_data = (nnl2_float64*)dividend->data->data;
            nnl2_float64* divisor_data = (nnl2_float64*)divisor->data->data;
            nnl2_float64* dividend_grad_data = (nnl2_float64*)dividend->grad->data;
            nnl2_float64* divisor_grad_data = (nnl2_float64*)divisor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                if(dividend->requires_grad) {
                    dividend_grad_data[i] += out_grad_data[i] / divisor_data[i];
                }
                if(divisor->requires_grad) {
                    divisor_grad_data[i] += -out_grad_data[i] * dividend_data[i] / (divisor_data[i] * divisor_data[i]);
                }
            }
			
            break;
        }
        
        case FLOAT32: {
			// Type-cast
            nnl2_float32* out_grad_data = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* dividend_data = (nnl2_float32*)dividend->data->data;
            nnl2_float32* divisor_data = (nnl2_float32*)divisor->data->data;
            nnl2_float32* dividend_grad_data = (nnl2_float32*)dividend->grad->data;
            nnl2_float32* divisor_grad_data = (nnl2_float32*)divisor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                if(dividend->requires_grad) {
                    dividend_grad_data[i] += out_grad_data[i] / divisor_data[i];
                }
                if(divisor->requires_grad) {
                    divisor_grad_data[i] += -out_grad_data[i] * dividend_data[i] / (divisor_data[i] * divisor_data[i]);
                }
            }
			
            break;
        }
        
        case INT32: {
			// Type-cast
            nnl2_float32* out_grad_data = (nnl2_float32*)out_tensor->grad->data;
            nnl2_int32* dividend_data = (nnl2_int32*)dividend->data->data;
            nnl2_int32* divisor_data = (nnl2_int32*)divisor->data->data;
            nnl2_float32* dividend_grad_data = (nnl2_float32*)dividend->grad->data;
            nnl2_float32* divisor_grad_data = (nnl2_float32*)divisor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                if(dividend->requires_grad && divisor_data[i] != 0) {
                    dividend_grad_data[i] += out_grad_data[i] / (nnl2_float32)divisor_data[i];
                }
                if(divisor->requires_grad && divisor_data[i] != 0) {
                    nnl2_float32 divisor_sq = (nnl2_float32)(divisor_data[i] * divisor_data[i]);
                    divisor_grad_data[i] += -out_grad_data[i] * (nnl2_float32)dividend_data[i] / divisor_sq;
                }
            }
			
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_DIV_BACKWARD_H **/
