#ifndef NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H

/** @file nnl2_ad_correspondence_pow_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence power operation
 **/

/** @brief 
 * Computes the gradient of the correspondence power operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence power operation
 *
 ** @param exponent_tensor 
 * The exponent input tensor to the correspondence power operation
 *
 ** @param exponent
 * The exponent value pointer
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_pow_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* exponent_tensor, void* exponent) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!exponent_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_pow_correspondence because exponent_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_pow_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent_tensor, "In function nnl2_ad_reverse_derivative_pow_correspondence, exponent_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "In function nnl2_ad_reverse_derivative_pow_correspondence, exponent is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_pow_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent_tensor->data, "In function nnl2_ad_reverse_derivative_pow_correspondence, exponent_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_pow_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent_tensor->data->shape, "In function nnl2_ad_reverse_derivative_pow_correspondence, exponent_tensor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = exponent_tensor->grad;
    nnl2_tensor* data_in  = exponent_tensor->data;

    switch(data_in->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64 k = *((nnl2_float64*)exponent);
            nnl2_float64* x = (nnl2_float64*)data_in->data;
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gin  = (nnl2_float64*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += gout[i] * k * pow(x[i], k - 1.0);
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32 k = *((nnl2_float32*)exponent);
            nnl2_float32* x = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += gout[i] * k * powf(x[i], k - 1.0f);
            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32 k = *((nnl2_int32*)exponent);
            nnl2_int32* x = (nnl2_int32*)data_in->data;
            nnl2_int32* gout = (nnl2_int32*)grad_out->data;
            nnl2_int32* gin  = (nnl2_int32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
			
            for(size_t i = 0; i < n; i++) {
                nnl2_int32 p = 1;
                for(int j = 0; j < k - 1; j++) p *= x[i];
                gin[i] += gout[i] * k * p;
            }
			
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

#endif /** NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H **/
