#ifndef NNL2_AD_ABS_BACKWARD_H
#define NNL2_AD_ABS_BACKWARD_H

/** @file nnl2_ad_abs_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for absolute value operation
 **/
 
/** @brief 
 * Computes the gradient of the absolute value operation in reverse mode AD 
 *
 ** @param out_tensor 
 * The output tensor from the absolute value operation
 *
 ** @param ad_tensor 
 * The input tensor to the absolute value operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP 
 *
 ** @see nnl2_product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_abs(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_abs because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_abs, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_abs, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_abs, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_abs, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_abs, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_abs, ad_tensor shape is NULL");
	#endif
    
	// Calculate total number of elements in tensors
    size_t numel = nnl2_product(out_tensor->data->shape, out_tensor->data->rank);
    
    switch(ad_tensor->data->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* out_grad_data = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* input_data = (nnl2_float64*)ad_tensor->data->data;
            nnl2_float64* input_grad_data = (nnl2_float64*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 sign;
                
				// Determine the sign for gradient propagation
                if(input_data[i] > 0) {
                    sign = 1.0;   // Derivative is 1
                } else if (input_data[i] < 0) {
                    sign = -1.0;  // Derivative is -1
                } else {
                    sign = 0.0;   // Derivative is 0
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        case FLOAT32: {
			// Type-cast
            nnl2_float32* out_grad_data = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* input_data = (nnl2_float32*)ad_tensor->data->data;
            nnl2_float32* input_grad_data = (nnl2_float32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 sign;
                
				// Determine the sign for gradient propagation
                if(input_data[i] > 0) {
                    sign = 1.0f;   // Derivative is 1
                } else if (input_data[i] < 0) {
                    sign = -1.0f;  // Derivative is -1
                } else {
                    sign = 0.0f;   // Derivative is 0
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        case INT32: {
			// Type-cast
            nnl2_int32* out_grad_data = (nnl2_int32*)out_tensor->grad->data;
            nnl2_int32* input_data = (nnl2_int32*)ad_tensor->data->data;
            nnl2_int32* input_grad_data = (nnl2_int32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_int32 sign;
                
				// Determine the sign for gradient propagation
                if(input_data[i] > 0) {
                    sign = 1;   // Derivative is 1
                } else if (input_data[i] < 0) {
                    sign = -1;  // Derivative is -1
                } else {
                    sign = 0;   // Derivative is 0
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(ad_tensor->data->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ABS_BACKWARD_H **/
