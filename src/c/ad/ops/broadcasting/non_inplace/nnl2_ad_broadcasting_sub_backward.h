#ifndef NNL2_AD_SUB_BROADCASTING_BACKWARD_H
#define NNL2_AD_SUB_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_sub_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting subtraction operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting subtraction operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting subtraction operation
 *
 ** @param minuend 
 * The minuend input tensor to the broadcasting subtraction operation
 *
 ** @param subtrahend 
 * The subtrahend input tensor to the broadcasting subtraction operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see nnl2_product
 ** @see nnl2_add_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_sub_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_sub_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend, "In function nnl2_ad_reverse_derivative_sub_broadcasting, minuend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend, "In function nnl2_ad_reverse_derivative_sub_broadcasting, subtrahend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_sub_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data, "In function nnl2_ad_reverse_derivative_sub_broadcasting, minuend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data, "In function nnl2_ad_reverse_derivative_sub_broadcasting, subtrahend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_sub_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(minuend->data->shape, "In function nnl2_ad_reverse_derivative_sub_broadcasting, minuend data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(subtrahend->data->shape, "In function nnl2_ad_reverse_derivative_sub_broadcasting, subtrahend data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_min = minuend->grad;
    nnl2_tensor* grad_sub = subtrahend->grad;
    nnl2_tensor* data_sub = subtrahend->data;

    size_t numel_out = nnl2_product(grad_out->shape, grad_out->rank);
    size_t numel_sub = nnl2_product(data_sub->shape, data_sub->rank);
    size_t blocks = numel_out / numel_sub;

    if(minuend->requires_grad) nnl2_add_inplace(grad_min, grad_out);

    if(subtrahend->requires_grad) {
        switch(data_sub->dtype) {
            case FLOAT64: {
				// Type-cast
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gsub = (nnl2_float64*)grad_sub->data;

                for(size_t i = 0; i < numel_sub; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            case FLOAT32: {
				// Type-cast
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gsub = (nnl2_float32*)grad_sub->data;

                for(size_t i = 0; i < numel_sub; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            case INT32: {
				// Type-cast
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gsub = (nnl2_int32*)grad_sub->data;

                for(size_t i = 0; i < numel_sub; i++) {
                    nnl2_int32 acc = 0;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            default: {
                NNL2_TYPE_ERROR(data_sub->dtype);
                break;
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_SUB_BROADCASTING_BACKWARD_H **/
