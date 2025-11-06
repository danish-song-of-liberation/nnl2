#ifndef NNL2_AD_POW_BACKWARD_H
#define NNL2_AD_POW_BACKWARD_H

/** @file nnl2_ad_pow_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for power operation
 **/

/** @brief 
 * Computes the gradient of the power operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the power operation
 *
 ** @param base 
 * The base input tensor to the power operation
 *
 ** @param exponent 
 * The exponent input tensor to the power operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_pow(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* base, nnl2_ad_tensor* exponent) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_pow, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base, "In function nnl2_ad_reverse_derivative_pow, base is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "In function nnl2_ad_reverse_derivative_pow, exponent is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_pow, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base->data, "In function nnl2_ad_reverse_derivative_pow, base data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data, "In function nnl2_ad_reverse_derivative_pow, exponent data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_pow, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base->data->shape, "In function nnl2_ad_reverse_derivative_pow, base data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data->shape, "In function nnl2_ad_reverse_derivative_pow, exponent data shape is NULL");
	#endif
    
    size_t numel = product(base->data->shape, base->data->rank);
    nnl2_tensor_type dtype = base->data->dtype;
	
	switch(dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* a_data = (nnl2_float64*)base->data->data;
            nnl2_float64* b_data = (nnl2_float64*)exponent->data->data;
            nnl2_float64* out_grad = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* a_grad = (nnl2_float64*)base->grad->data;
            nnl2_float64* b_grad = (nnl2_float64*)exponent->grad->data;

            if(base->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0.0) a_grad[i] += out_grad[i] * b_data[i] * pow(a_data[i], b_data[i] - 1.0);
                }
            }

            if(exponent->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] > 0.0) b_grad[i] += out_grad[i] * pow(a_data[i], b_data[i]) * log(a_data[i]);
                }
            }
			
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* a_data = (nnl2_float32*)base->data->data;
            nnl2_float32* b_data = (nnl2_float32*)exponent->data->data;
            nnl2_float32* out_grad = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* a_grad = (nnl2_float32*)base->grad->data;
            nnl2_float32* b_grad = (nnl2_float32*)exponent->grad->data;

            if(base->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0.0f) a_grad[i] += out_grad[i] * b_data[i] * powf(a_data[i], b_data[i] - 1.0f);
                }
            }

            if(exponent->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] > 0.0f) b_grad[i] += out_grad[i] * powf(a_data[i], b_data[i]) * logf(a_data[i]);
                }
            }
			
            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32* a_data = (nnl2_int32*)base->data->data;
            nnl2_int32* b_data = (nnl2_int32*)exponent->data->data;
            nnl2_float32* out_grad = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* a_grad = (nnl2_float32*)base->grad->data;
            nnl2_float32* b_grad = (nnl2_float32*)exponent->grad->data;

            if(base->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0) a_grad[i] += out_grad[i] * b_data[i] * powf((float)a_data[i], (float)b_data[i] - 1.0f);
                }
            }

            if(exponent->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] > 0) b_grad[i] += out_grad[i] * powf((float)a_data[i], (float)b_data[i]) * logf((float)a_data[i]);
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

#endif /** NNL2_AD_POW_BACKWARD_H **/
