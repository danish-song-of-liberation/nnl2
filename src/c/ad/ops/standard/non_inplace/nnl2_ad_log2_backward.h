#ifndef NNL2_AD_LOG2_BACKWARD_DERIVATIVE_H
#define NNL2_AD_LOG2_BACKWARD_DERIVATIVE_H

/** @file nnl2_ad_log2_backward_derivative.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for base-2 logarithm operation
 **/

/** @brief 
 * Computes the gradient of the base-2 logarithm operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the base-2 logarithm operation
 *
 ** @param ad_tensor 
 * The input tensor to the base-2 logarithm operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Derivative: d(log2(x))/dx = 1/(x * ln(2)) = (1/x) * (1/ln(2))
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_log2(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_log2 because ad_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_log2, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_log2, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_log2, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_log2, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_log2, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_log2, ad_tensor shape is NULL");
	#endif
    
	// Calculate total number of elements in tensors
    size_t numel = product(ad_tensor->data->shape, ad_tensor->data->rank);
    nnl2_tensor_type dtype = ad_tensor->data->dtype;

    switch(dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* xd = (nnl2_float64*)ad_tensor->data->data;
            nnl2_float64* yg = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* xg = (nnl2_float64*)ad_tensor->grad->data;
			
			const nnl2_float64 ln2 = 0.69314718055994530942; // ln(2)
            
            for(size_t i = 0; i < numel; i++) {
                xg[i] += yg[i] / (xd[i] * ln2);
            }
			
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* xd = (nnl2_float32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
			
			const nnl2_float32 ln2 = 0.69314718055994530942f; // ln(2)
            
            for(size_t i = 0; i < numel; i++) {
                xg[i] += yg[i] / (xd[i] * ln2);
            }
			
            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32* xd = (nnl2_int32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
			
			const nnl2_float32 ln2 = 0.69314718055994530942f; // ln(2)
            
            for(size_t i = 0; i < numel; i++) {
                if(xd[i] != 0) {
                    xg[i] += yg[i] / ((nnl2_float32)xd[i] * ln2);
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

#endif /** NNL2_AD_LOG2_BACKWARD_DERIVATIVE_H **/
