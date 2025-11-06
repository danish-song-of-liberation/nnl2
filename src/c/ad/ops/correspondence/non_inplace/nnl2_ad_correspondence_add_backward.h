#ifndef NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H

/** @file nnl2_ad_correspondence_add_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence addition operation
 **/

/** @brief 
 * Computes the gradient of the correspondence addition operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence addition operation
 *
 ** @param inc_tensor 
 * The input tensor to the correspondence addition operation
 *
 ** @param inc
 * The increment value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see add_incf_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_add_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* inc_tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	(void)inc;
	(void)out_tensor;	
	
    if(!inc_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_add_correspondence because inc_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_add_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(inc_tensor, "In function nnl2_ad_reverse_derivative_add_correspondence, inc_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_add_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(inc_tensor->data, "In function nnl2_ad_reverse_derivative_add_correspondence, inc_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_add_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(inc_tensor->data->shape, "In function nnl2_ad_reverse_derivative_add_correspondence, inc_tensor data shape is NULL");
	#endif
    
	if(inc_tensor->requires_grad) {
		switch(inc_tensor->data->dtype) {
			case FLOAT64: add_incf_inplace(inc_tensor->grad, &(nnl2_float64){1.0});  break;
			case FLOAT32: add_incf_inplace(inc_tensor->grad, &(nnl2_float32){1.0f}); break;
			case INT32:   add_incf_inplace(inc_tensor->grad, &(nnl2_int32){1});      break;
			
			default: {
				NNL2_TYPE_ERROR(inc_tensor->data->dtype);
				break;
			}
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_ADD_BACKWARD_H **/
