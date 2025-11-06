#ifndef NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H

/** @file nnl2_ad_correspondence_div_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence division operation
 **/

/** @brief 
 * Computes the gradient of the correspondence division operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence division operation
 *
 ** @param divisor_tensor 
 * The divisor input tensor to the correspondence division operation
 *
 ** @param divisor
 * The divisor value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see axpy_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_div_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* divisor_tensor, void* divisor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!divisor_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_div_correspondence because divisor_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_div_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor_tensor, "In function nnl2_ad_reverse_derivative_div_correspondence, divisor_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "In function nnl2_ad_reverse_derivative_div_correspondence, divisor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_div_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor_tensor->data, "In function nnl2_ad_reverse_derivative_div_correspondence, divisor_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_div_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor_tensor->data->shape, "In function nnl2_ad_reverse_derivative_div_correspondence, divisor_tensor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = divisor_tensor->grad;

    switch(divisor_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64 c = *((nnl2_float64*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0 / c);
            break;
        }
		
        case FLOAT32: {
            nnl2_float32 c = *((nnl2_float32*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0f / c);
            break;
        }
		
        case INT32: {
            nnl2_int32 c = *((nnl2_int32*)divisor);
            axpy_inplace(grad_in, grad_out, 1.0f / (float)c);
            break;
        }
		
        default: {
            NNL2_TYPE_ERROR(divisor_tensor->data->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_DIV_BACKWARD_H **/
