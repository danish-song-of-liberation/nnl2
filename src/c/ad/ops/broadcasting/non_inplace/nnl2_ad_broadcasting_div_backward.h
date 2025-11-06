#ifndef NNL2_AD_DIV_BROADCASTING_BACKWARD_H
#define NNL2_AD_DIV_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_div_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting division operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting division operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting division operation
 *
 ** @param dividend 
 * The dividend input tensor to the broadcasting division operation
 *
 ** @param divisor 
 * The divisor input tensor to the broadcasting division operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_div_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* dividend, nnl2_ad_tensor* divisor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_div_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend, "In function nnl2_ad_reverse_derivative_div_broadcasting, dividend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "In function nnl2_ad_reverse_derivative_div_broadcasting, divisor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_div_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data, "In function nnl2_ad_reverse_derivative_div_broadcasting, dividend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data, "In function nnl2_ad_reverse_derivative_div_broadcasting, divisor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_div_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(dividend->data->shape, "In function nnl2_ad_reverse_derivative_div_broadcasting, dividend data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(divisor->data->shape, "In function nnl2_ad_reverse_derivative_div_broadcasting, divisor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_dividend = dividend->grad;
    nnl2_tensor* grad_divisor = divisor->grad;

    nnl2_tensor* data_dividend = dividend->data;
    nnl2_tensor* data_divisor = divisor->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_dividend = product(data_dividend->shape, data_dividend->rank);
    size_t numel_divisor = product(data_divisor->shape, data_divisor->rank);

    size_t blocks_dividend = numel_out / numel_dividend;
    size_t blocks_divisor = numel_out / numel_divisor;

    switch(data_dividend->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gdivd = (nnl2_float64*)grad_dividend->data;
            nnl2_float64* gdivs = (nnl2_float64*)grad_divisor->data;
            nnl2_float64* divd_data = (nnl2_float64*)data_dividend->data;
            nnl2_float64* divs_data = (nnl2_float64*)data_divisor->data;

            if(dividend->requires_grad) {
                for(size_t i = 0; i < numel_dividend; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks_dividend; b++) {
                        size_t j = b * numel_dividend + i;
                        size_t divs_idx = (b * numel_divisor) % numel_divisor;
                        if(divs_data[divs_idx] != 0.0) {
                            acc += gout[j] / divs_data[divs_idx];
                        }
                    }
                    gdivd[i] += acc;
                }
            }

            if(divisor->requires_grad) {
                for(size_t i = 0; i < numel_divisor; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks_divisor; b++) {
                        size_t j = b * numel_divisor + i;
                        size_t divd_idx = (b * numel_dividend) % numel_dividend;
                        if(divs_data[i] != 0.0) {
                            acc += -gout[j] * divd_data[divd_idx] / (divs_data[i] * divs_data[i]);
                        }
                    }
					
                    gdivs[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gdivd = (nnl2_float32*)grad_dividend->data;
            nnl2_float32* gdivs = (nnl2_float32*)grad_divisor->data;
            nnl2_float32* divd_data = (nnl2_float32*)data_dividend->data;
            nnl2_float32* divs_data = (nnl2_float32*)data_divisor->data;

            if(dividend->requires_grad) {
                for(size_t i = 0; i < numel_dividend; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_dividend; b++) {
                        size_t j = b * numel_dividend + i;
                        size_t divs_idx = (b * numel_divisor) % numel_divisor;
                        if(divs_data[divs_idx] != 0.0f) {
                            acc += gout[j] / divs_data[divs_idx];
                        }
                    }
					
                    gdivd[i] += acc;
                }
            }

            if(divisor->requires_grad) {
                for(size_t i = 0; i < numel_divisor; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_divisor; b++) {
                        size_t j = b * numel_divisor + i;
                        size_t divd_idx = (b * numel_dividend) % numel_dividend;
                        if(divs_data[i] != 0.0f) {
                            acc += -gout[j] * divd_data[divd_idx] / (divs_data[i] * divs_data[i]);
                        }
                    }
					
                    gdivs[i] += acc;
                }
            }

            break;
        }

        case INT32: {
			// Type-cast
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gdivd = (nnl2_float32*)grad_dividend->data;
            nnl2_float32* gdivs = (nnl2_float32*)grad_divisor->data;
            nnl2_int32* divd_data = (nnl2_int32*)data_dividend->data;
            nnl2_int32* divs_data = (nnl2_int32*)data_divisor->data;

            if(dividend->requires_grad) {
                for(size_t i = 0; i < numel_dividend; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_dividend; b++) {
                        size_t j = b * numel_dividend + i;
                        size_t divs_idx = (b * numel_divisor) % numel_divisor;
                        if(divs_data[divs_idx] != 0) {
                            acc += gout[j] / (nnl2_float32)divs_data[divs_idx];
                        }
                    }
					
                    gdivd[i] += acc;
                }
            }

            if(divisor->requires_grad) {
                for(size_t i = 0; i < numel_divisor; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_divisor; b++) {
                        size_t j = b * numel_divisor + i;
                        size_t divd_idx = (b * numel_dividend) % numel_dividend;
                        if(divs_data[i] != 0) {
                            nnl2_float32 divs_sq = (nnl2_float32)(divs_data[i] * divs_data[i]);
                            acc += -gout[j] * (nnl2_float32)divd_data[divd_idx] / divs_sq;
                        }
                    }
					
                    gdivs[i] += acc;
                }
            }

            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_dividend->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_DIV_BROADCASTING_BACKWARD_H **/
