#ifndef NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H

/** @file nnl2_ad_correspondence_mul_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence multiplication operation
 **/

/** @brief 
 * Computes the gradient of the correspondence multiplication operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence multiplication operation
 *
 ** @param multiplier_tensor 
 * The multiplier input tensor to the correspondence multiplication operation
 *
 ** @param multiplier
 * The multiplier value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see axpy_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_mul_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* multiplier_tensor, void* multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!multiplier_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_mul_correspondence because multiplier_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_mul_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier_tensor, "In function nnl2_ad_reverse_derivative_mul_correspondence, multiplier_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "In function nnl2_ad_reverse_derivative_mul_correspondence, multiplier is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_mul_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier_tensor->data, "In function nnl2_ad_reverse_derivative_mul_correspondence, multiplier_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mul_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier_tensor->data->shape, "In function nnl2_ad_reverse_derivative_mul_correspondence, multiplier_tensor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = multiplier_tensor->grad;

    switch(multiplier_tensor->data->dtype) {
		case FLOAT64: axpy_inplace(grad_in, grad_out, *((nnl2_float64*)multiplier)); break;
		case FLOAT32: axpy_inplace(grad_in, grad_out, *((nnl2_float32*)multiplier)); break;
		case INT32:   axpy_inplace(grad_in, grad_out, *((nnl2_int32*)multiplier));   break; 
		
		default: {
			NNL2_TYPE_ERROR(multiplier_tensor->data->dtype);
			break;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_MUL_BACKWARD_H **/
