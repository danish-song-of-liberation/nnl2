#ifndef NNL2_AD_MIN_BROADCASTING_BACKWARD_H
#define NNL2_AD_MIN_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_min_broadcasting_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting minimum operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting minimum operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting minimum operation
 *
 ** @param a 
 * The first input tensor to the broadcasting minimum operation
 *
 ** @param b 
 * The second input tensor to the broadcasting minimum operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_min_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, nnl2_ad_tensor* b) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_min_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_min_broadcasting, a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b, "In function nnl2_ad_reverse_derivative_min_broadcasting, b is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_min_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_min_broadcasting, a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b->data, "In function nnl2_ad_reverse_derivative_min_broadcasting, b data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_min_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data->shape, "In function nnl2_ad_reverse_derivative_min_broadcasting, a data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(b->data->shape, "In function nnl2_ad_reverse_derivative_min_broadcasting, b data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* data_a = a->data;
    nnl2_tensor* data_b = b->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_a = product(data_a->shape, data_a->rank);
    size_t numel_b = product(data_b->shape, data_b->rank);

    size_t blocks_a = numel_out / numel_a;
    size_t blocks_b = numel_out / numel_b;

    switch(data_a->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* ad = (nnl2_float64*)data_a->data;
            nnl2_float64* bd = (nnl2_float64*)data_b->data;
            nnl2_float64* gd = (nnl2_float64*)grad_out->data;
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;
            nnl2_float64* bg = (nnl2_float64*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel_a; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if(ad[i] < bd[j % numel_b]) acc += gd[j];
                    }
					
                    ag[i] += acc;
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel_b; i++) {
                    nnl2_float64 acc = 0.0;
                    for(size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if(bd[i] < ad[j % numel_a]) acc += gd[j];
                    }
					
                    bg[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32* ad = (nnl2_float32*)data_a->data;
            nnl2_float32* bd = (nnl2_float32*)data_b->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            nnl2_float32* bg = (nnl2_float32*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel_a; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if(ad[i] < bd[j % numel_b]) acc += gd[j];
                    }
                    ag[i] += acc;
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel_b; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if(bd[i] < ad[j % numel_a]) acc += gd[j];
                    }
					
                    bg[i] += acc;
                }
            }

            break;
        }

        case INT32: {
			// Type-cast
            nnl2_int32* ad = (nnl2_int32*)data_a->data;
            nnl2_int32* bd = (nnl2_int32*)data_b->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            nnl2_float32* bg = (nnl2_float32*)b->grad->data;

            if(a->requires_grad) {
                for(size_t i = 0; i < numel_a; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if(ad[i] < bd[j % numel_b]) acc += gd[j];
                    }
                    ag[i] += acc;
                }
            }

            if(b->requires_grad) {
                for(size_t i = 0; i < numel_b; i++) {
                    nnl2_float32 acc = 0.0f;
                    for(size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if(bd[i] < ad[j % numel_a]) acc += gd[j];
                    }
					
                    bg[i] += acc;
                }
            }

            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_a->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_MIN_BROADCASTING_BACKWARD_H **/
