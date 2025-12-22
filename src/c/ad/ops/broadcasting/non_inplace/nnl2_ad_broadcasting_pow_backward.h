#ifndef NNL2_AD_POW_BROADCASTING_BACKWARD_H
#define NNL2_AD_POW_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_pow_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting power operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting power operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting power operation
 *
 ** @param base 
 * The base input tensor to the broadcasting power operation
 *
 ** @param exponent 
 * The exponent input tensor to the broadcasting power operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_pow_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* base, nnl2_ad_tensor* exponent) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_pow_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base, "In function nnl2_ad_reverse_derivative_pow_broadcasting, base is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent, "In function nnl2_ad_reverse_derivative_pow_broadcasting, exponent is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_pow_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base->data, "In function nnl2_ad_reverse_derivative_pow_broadcasting, base data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data, "In function nnl2_ad_reverse_derivative_pow_broadcasting, exponent data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_pow_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(base->data->shape, "In function nnl2_ad_reverse_derivative_pow_broadcasting, base data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(exponent->data->shape, "In function nnl2_ad_reverse_derivative_pow_broadcasting, exponent data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_base = base->grad;
    nnl2_tensor* grad_exp = exponent->grad;
    nnl2_tensor* data_base = base->data;
    nnl2_tensor* data_exp = exponent->data;

    size_t numel_out = nnl2_product(grad_out->shape, grad_out->rank);
    size_t numel_base = nnl2_product(data_base->shape, data_base->rank);
    size_t numel_exp = nnl2_product(data_exp->shape, data_exp->rank);
    size_t blocks_for_base = numel_out / numel_base;
    size_t blocks_for_exp = numel_out / numel_exp;

    if(base->requires_grad || exponent->requires_grad) {
        switch(data_base->dtype) {
            case FLOAT64: {
				// Type-cast
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gbase = (nnl2_float64*)grad_base->data;
                nnl2_float64* gexp = (nnl2_float64*)grad_exp->data;
                nnl2_float64* bbase = (nnl2_float64*)data_base->data;
                nnl2_float64* bexp = (nnl2_float64*)data_exp->data;

                if(base->requires_grad) {
                    for(size_t i = 0; i < numel_base; i++) {
                        nnl2_float64 acc = 0.0;
                        for(size_t b = 0; b < blocks_for_base; b++) {
                            size_t idx = b * numel_base + i;
                            nnl2_float64 a = bbase[i];
                            nnl2_float64 e = bexp[idx % numel_exp];
                            acc += gout[idx] * e * pow(a, e - 1.0);
                        }
						
                        gbase[i] += acc;
                    }
                }

                if(exponent->requires_grad) {
                    for(size_t i = 0; i < numel_exp; i++) {
                        nnl2_float64 acc = 0.0;
                        for(size_t b = 0; b < blocks_for_exp; b++) {
                            size_t idx = b * numel_exp + i;
                            nnl2_float64 a = bbase[idx % numel_base];
                            nnl2_float64 e = bexp[i];
                            acc += gout[idx] * pow(a, e) * log(a);
                        }
						
                        gexp[i] += acc;
                    }
                }
				
                break;
            }

            case FLOAT32: {
				// Type-cast
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gbase = (nnl2_float32*)grad_base->data;
                nnl2_float32* gexp = (nnl2_float32*)grad_exp->data;
                nnl2_float32* bbase = (nnl2_float32*)data_base->data;
                nnl2_float32* bexp = (nnl2_float32*)data_exp->data;

                if(base->requires_grad) {
                    for(size_t i = 0; i < numel_base; i++) {
                        nnl2_float32 acc = 0.0f;
                        for(size_t b = 0; b < blocks_for_base; b++) {
                            size_t idx = b * numel_base + i;
                            nnl2_float32 a = bbase[i];
                            nnl2_float32 e = bexp[idx % numel_exp];
                            acc += gout[idx] * e * powf(a, e - 1.0f);
                        }
						
                        gbase[i] += acc;
                    }
                }

                if(exponent->requires_grad) {
                    for(size_t i = 0; i < numel_exp; i++) {
                        nnl2_float32 acc = 0.0f;
                        for(size_t b = 0; b < blocks_for_exp; b++) {
                            size_t idx = b * numel_exp + i;
                            nnl2_float32 a = bbase[idx % numel_base];
                            nnl2_float32 e = bexp[i];
                            acc += gout[idx] * powf(a, e) * logf(a);
                        }
						
                        gexp[i] += acc;
                    }
                }
				
                break;
            }

            default: {
                NNL2_TYPE_ERROR(data_base->dtype);
                break;
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_POW_BROADCASTING_BACKWARD_H **/
