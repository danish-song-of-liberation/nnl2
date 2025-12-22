#ifndef NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H

/** @file nnl2_ad_correspondence_min_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence minimum operation
 **/

/** @brief 
 * Computes the gradient of the correspondence minimum operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence minimum operation
 *
 ** @param threshold_tensor 
 * The threshold input tensor to the correspondence minimum operation
 *
 ** @param threshold
 * The threshold value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_min_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* threshold_tensor, void* threshold) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!threshold_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_min_correspondence because threshold_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_min_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(threshold_tensor, "In function nnl2_ad_reverse_derivative_min_correspondence, threshold_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(threshold, "In function nnl2_ad_reverse_derivative_min_correspondence, threshold is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_min_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(threshold_tensor->data, "In function nnl2_ad_reverse_derivative_min_correspondence, threshold_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_min_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(threshold_tensor->data->shape, "In function nnl2_ad_reverse_derivative_min_correspondence, threshold_tensor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = threshold_tensor->grad;
    nnl2_tensor* data_in  = threshold_tensor->data;

    switch(data_in->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64 th = *((nnl2_float64*)threshold);
            nnl2_float64* x = (nnl2_float64*)data_in->data;
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gin  = (nnl2_float64*)grad_in->data;
            size_t n = nnl2_product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0.0;
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32 th = *((nnl2_float32*)threshold);
            nnl2_float32* x = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            size_t n = nnl2_product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0.0f;
            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32 th = *((nnl2_int32*)threshold);
            nnl2_int32* x = (nnl2_int32*)data_in->data;
            nnl2_int32* gout = (nnl2_int32*)grad_out->data;
            nnl2_int32* gin  = (nnl2_int32*)grad_in->data;
            size_t n = nnl2_product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0;
            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_in->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H **/
