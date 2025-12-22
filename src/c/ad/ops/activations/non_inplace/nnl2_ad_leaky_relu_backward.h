#ifndef NNL2_AD_LEAKYRELU_BACKWARD_H
#define NNL2_AD_LEAKYRELU_BACKWARD_H

// NNL2

/** @file nnl2_ad_leakyrelu_backward.h
 ** @brief Reverse mode derivative implementation for LeakyReLU operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Compute derivative of LeakyReLU operation and propagate gradients
 *
 ** @param out_tensor
 * Output tensor from LeakyReLU forward pass
 *
 ** @param a
 * Input tensor to LeakyReLU operation
 *
 ** @param alpha
 * Negative slope coefficient for x < 0
 *
 ** @details
 * Derivative of LeakyReLU(x) is: 1 if x >= 0, alpha if x < 0
 * Accumulates gradients in input tensor's gradient buffer
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
 ** @see nnl2_ad_leakyrelu()
 ** @see nnl2_ad_reverse_backward_leakyrelu()
 **/
void nnl2_ad_reverse_derivative_leakyrelu(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, nnl2_float32 alpha) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_leakyrelu, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_leakyrelu, input tensor a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_leakyrelu, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_leakyrelu, input tensor a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_leakyrelu, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->grad, "In function nnl2_ad_reverse_derivative_leakyrelu, input tensor a grad is NULL");
	#endif
	
    size_t numel = nnl2_product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)a->data->data;         
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data; 
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;       
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0.0) ? gd[i] : (nnl2_float64)(alpha * gd[i]);
			}
            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0.0f) ? gd[i] : (alpha * gd[i]);
			}
            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0) ? gd[i] : (alpha * gd[i]);
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

#endif /** NNL2_AD_LEAKYRELU_BACKWARD_H **/
