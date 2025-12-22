#ifndef NNL2_AD_SIGMOID_BACKWARD_H
#define NNL2_AD_SIGMOID_BACKWARD_H

// NNL2

/** @file nnl2_ad_sigmoid_backward.h
 ** @brief Reverse mode derivative implementation for sigmoid operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Compute derivative of sigmoid operation and propagate gradients
 *
 ** @param out_tensor
 * Output tensor from sigmoid forward pass
 *
 ** @param a
 * Input tensor to sigmoid operation
 *
 ** @param approx
 * Whether to use approximate sigmoid calculation
 *
 ** @details
 * Derivative of sigmoid(x) is: sigmoid(x) * (1 - sigmoid(x))
 * Uses either exact exponential or fast approximation based on approx parameter
 *
 ** @exception NNL2Error
 * If any tensor pointer is NULL and safety mode is MAX
 *
 ** @exception NNL2Error  
 * If any tensor data pointer is NULL and safety mode is MAX
 *
 ** @exception NNL2Error
 * If tensor shapes are incompatible
 *
 ** @see nnl2_ad_sigmoid()
 ** @see nnl2_ad_reverse_backward_sigmoid()
 **/
void nnl2_ad_reverse_derivative_sigmoid(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_sigmoid, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_sigmoid, input tensor a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_sigmoid, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_sigmoid, input tensor a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_sigmoid, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->grad, "In function nnl2_ad_reverse_derivative_sigmoid, input tensor a grad is NULL");
	#endif
	
    size_t numel = nnl2_product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)a->data->data;
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 s;
                if(approx) {
                    nnl2_float64 x = ad[i];
                    nnl2_float64 abs_x = fabs(x);
                    s = 0.5 + 0.5 * x / (1.0 + abs_x);
                } else {
                    s = 1.0 / (1.0 + exp(-ad[i]));
                }
                ag[i] += gd[i] * s * (1.0 - s);
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 s;
                if(approx) {
                    nnl2_float32 x = ad[i];
                    if (x < -8.0f) s = 0.0f;
                    else if (x > 8.0f) s = 1.0f;
                    else {
                        nnl2_float32 abs_x = fabsf(x);
                        s = 0.5f + 0.5f * x / (1.0f + abs_x);
                    }
                } else {
                    s = 1.0f / (1.0f + expf(-ad[i]));
                }
                ag[i] += gd[i] * s * (1.0f - s);
            }
			
            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 x = (nnl2_float64)ad[i];
                nnl2_float64 s;
                if(approx) {
                    nnl2_float64 abs_x = fabs(x);
                    s = 0.5 + 0.5 * x / (1.0 + abs_x);
                } else {
                    s = 1.0 / (1.0 + exp(-x));
                }
                ag[i] += gd[i] * (nnl2_float32)(s * (1.0 - s));
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

#endif /** NNL2_AD_SIGMOID_BACKWARD_H **/
