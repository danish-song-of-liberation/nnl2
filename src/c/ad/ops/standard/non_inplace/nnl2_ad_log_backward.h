#ifndef NNL2_AD_LOG_BACKWARD_DERIVATIVE_H
#define NNL2_AD_LOG_BACKWARD_DERIVATIVE_H

/** @file nnl2_ad_log_backward_derivative.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for natural logarithm operation
 **/

/** @brief 
 * Computes the gradient of the natural logarithm operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the logarithm operation
 *
 ** @param ad_tensor 
 * The input tensor to the logarithm operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_log(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_log because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_log, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_log, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_log, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_log, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_log, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_log, ad_tensor shape is NULL");
	#endif
    
	// Calculate total number of elements in tensors
    size_t numel = nnl2_product(ad_tensor->data->shape, ad_tensor->data->rank);
    nnl2_tensor_type dtype = ad_tensor->data->dtype;

    switch(dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* xd = (nnl2_float64*)ad_tensor->data->data;
            nnl2_float64* yg = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* xg = (nnl2_float64*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                xg[i] += yg[i] / xd[i];
            }
			
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* xd = (nnl2_float32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                xg[i] += yg[i] / xd[i];
            }
			
            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32* xd = (nnl2_int32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                if(xd[i] != 0) {
                    xg[i] += yg[i] / (nnl2_float32)xd[i];
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

#endif /** NNL2_AD_LOG_BACKWARD_DERIVATIVE_H **/
