#ifndef NNL2_AD_MAX_BACKWARD_H
#define NNL2_AD_MAX_BACKWARD_H

/** @file nnl2_ad_max_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for maximum operation
 **/

/** @brief 
 * Computes the gradient of the maximum operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the maximum operation
 *
 ** @param a 
 * The first input tensor to the maximum operation
 *
 ** @param b 
 * The second input tensor to the maximum operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_max(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, nnl2_ad_tensor* b) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_max, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_max, a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b, "In function nnl2_ad_reverse_derivative_max, b is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_max, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_max, a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b->data, "In function nnl2_ad_reverse_derivative_max, b data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_max, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data->shape, "In function nnl2_ad_reverse_derivative_max, a data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b->data->shape, "In function nnl2_ad_reverse_derivative_max, b data shape is NULL");
	#endif
    
    size_t numel = product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch(dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* ad = (nnl2_float64*)a->data->data;
            nnl2_float64* bd = (nnl2_float64*)b->data->data;
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;
            nnl2_float64* bg = (nnl2_float64*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(ad[i] > bd[i]) ag[i] += gd[i];
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(bd[i] > ad[i]) bg[i] += gd[i];
                }
            }

            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* bd = (nnl2_float32*)b->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            nnl2_float32* bg = (nnl2_float32*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(ad[i] > bd[i]) ag[i] += gd[i];
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(bd[i] > ad[i]) bg[i] += gd[i];
                }
            }

            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_int32* bd = (nnl2_int32*)b->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            nnl2_float32* bg = (nnl2_float32*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(ad[i] > bd[i]) ag[i] += gd[i];
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel; i++) {
                    if(bd[i] > ad[i]) bg[i] += gd[i];
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

#endif /** NNL2_AD_MAX_BACKWARD_H **/
