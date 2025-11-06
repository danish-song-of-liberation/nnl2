#ifndef NNL2_AD_AXPY_BROADCASTING_BACKWARD_H
#define NNL2_AD_AXPY_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_axpy_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting axpy operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting axpy operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting axpy operation
 *
 ** @param axpyend 
 * The first input tensor to the broadcasting axpy operation
 *
 ** @param sumend 
 * The second input tensor to the broadcasting axpy operation
 *
 ** @param multiplier
 * The multiplier scalar value
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_axpy_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* axpyend, nnl2_ad_tensor* sumend, float multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(axpyend, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, axpyend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(axpyend->data, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, axpyend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(axpyend->data->shape, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, axpyend data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_axpy_broadcasting, sumend data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_axpy = axpyend->grad;
    nnl2_tensor* grad_sum = sumend->grad;

    nnl2_tensor* data_axpy = axpyend->data;
    nnl2_tensor* data_sum = sumend->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_axpy = product(data_axpy->shape, data_axpy->rank);
    size_t numel_sum = product(data_sum->shape, data_sum->rank);

    size_t blocks_axpy = numel_out / numel_axpy;
    size_t blocks_sum = numel_out / numel_sum;

    switch(data_axpy->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gax = (nnl2_float64*)grad_axpy->data;
            nnl2_float64* gsum = (nnl2_float64*)grad_sum->data;

            if(axpyend->requires_grad) {
                for(size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * (nnl2_float64)multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if(sumend->requires_grad) {
                for(size_t i = 0; i < numel_sum; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gax = (nnl2_float32*)grad_axpy->data;
            nnl2_float32* gsum = (nnl2_float32*)grad_sum->data;

            if(axpyend->requires_grad) {
                for(size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * (nnl2_float32)multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if(sumend->requires_grad) {
                for(size_t i = 0; i < numel_sum; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        case INT32: {
			// Type-cast
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gax = (nnl2_float32*)grad_axpy->data;
            nnl2_float32* gsum = (nnl2_float32*)grad_sum->data;

            if(axpyend->requires_grad) {
                for(size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if(sumend->requires_grad) {
                for(size_t i = 0; i < numel_sum; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_axpy->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_AXPY_BROADCASTING_BACKWARD_H **/
