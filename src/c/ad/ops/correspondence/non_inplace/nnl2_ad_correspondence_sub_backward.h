#ifndef NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H

/** @file nnl2_ad_correspondence_sub_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence subtraction operation
 **/

/** @brief 
 * Computes the gradient of the correspondence subtraction operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence subtraction operation
 *
 ** @param dec_tensor 
 * The decrement input tensor to the correspondence subtraction operation
 *
 ** @param dec
 * The decrement value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see add_incf_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sub_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* dec_tensor, void* dec) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	(void)dec;
	(void)out_tensor;	
	
    if(!dec_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_sub_correspondence because dec_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_sub_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dec_tensor, "In function nnl2_ad_reverse_derivative_sub_correspondence, dec_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_sub_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dec_tensor->data, "In function nnl2_ad_reverse_derivative_sub_correspondence, dec_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sub_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dec_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sub_correspondence, dec_tensor data shape is NULL");
	#endif
    
	if(dec_tensor->requires_grad) {
		switch(dec_tensor->data->dtype) {
			case FLOAT64: add_incf_inplace(dec_tensor->grad, &(nnl2_float64){1.0});  break;
			case FLOAT32: add_incf_inplace(dec_tensor->grad, &(nnl2_float32){1.0f}); break;
			case INT32:   add_incf_inplace(dec_tensor->grad, &(nnl2_int32){1});      break;
			
			default: {
				NNL2_TYPE_ERROR(dec_tensor->data->dtype);
				break;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_SUB_BACKWARD_H **/
