#ifndef NNL2_AD_ADD_BROADCASTING_BACKWARD_H
#define NNL2_AD_ADD_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_add_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting addition operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting addition operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting addition operation
 *
 ** @param summand 
 * The first input tensor to the broadcasting addition operation
 *
 ** @param sumend 
 * The second input tensor to the broadcasting addition operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 ** @see nnl2_add_inplace
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_add_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_add_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "In function nnl2_ad_reverse_derivative_add_broadcasting, summand is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_add_broadcasting, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_add_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "In function nnl2_ad_reverse_derivative_add_broadcasting, summand data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_add_broadcasting, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_add_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data->shape, "In function nnl2_ad_reverse_derivative_add_broadcasting, summand data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_add_broadcasting, sumend data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_sum = summand->grad;
    nnl2_tensor* grad_end = sumend->grad;
    nnl2_tensor* data_end = sumend->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_end = product(data_end->shape, data_end->rank);
    size_t blocks = numel_out / numel_end;
	
    if(summand->requires_grad) nnl2_add_inplace(grad_sum, grad_out);
	
	if(sumend->requires_grad) {
        switch(data_end->dtype) {
            case FLOAT64: {
				// Type-cast
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gend = (nnl2_float64*)grad_end->data;
				
                for(size_t i = 0; i < numel_end; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }
			
			case FLOAT32: {
				// Type-cast
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gend = (nnl2_float32*)grad_end->data;
				
                for(size_t i = 0; i < numel_end; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }
			
			case INT32: {
				// Type-cast
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gend = (nnl2_int32*)grad_end->data;
				
                for(size_t i = 0; i < numel_end; i++) {
                    nnl2_int32 acc = 0;
                    for(size_t b = 0; b < blocks; b++) acc += gout[b * numel_end + i];
                    gend[i] += acc;
                }
				
                break;
            }

            default: {
                NNL2_TYPE_ERROR(data_end->dtype);
                break;
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ADD_BROADCASTING_BACKWARD_H **/
