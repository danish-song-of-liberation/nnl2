#ifndef NNL2_AD_RELU_BACKWARD_H
#define NNL2_AD_RELU_BACKWARD_H

// NNL2

/** @file nnl2_ad_relu_backward.h
 ** @brief Reverse mode derivative implementation for ReLU operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Compute derivative of ReLU operation and propagate gradients
 *
 ** @param out_tensor
 * Output tensor from ReLU forward pass
 *
 ** @param a
 * Input tensor to ReLU operation
 *
 ** @details
 * Derivative of ReLU(x) is: 1 if x > 0, 0 if x <= 0
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
 ** @see nnl2_ad_relu()
 ** @see nnl2_ad_reverse_backward_relu()
 **/
void nnl2_ad_reverse_derivative_relu(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_relu, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_relu, input tensor a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_relu, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_relu, input tensor a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_relu, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->grad, "In function nnl2_ad_reverse_derivative_relu, input tensor a grad is NULL");
	#endif
	
    size_t numel = nnl2_product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)a->data->data;          
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data; 
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;          
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0.0) ? gd[i] : 0.0;
			}
            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0.0f) ? gd[i] : 0.0f;
			}
            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) {
				ag[i] += (ad[i] > 0) ? gd[i] : 0.0f;
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

#endif /** NNL2_AD_RELU_BACKWARD_H **/
