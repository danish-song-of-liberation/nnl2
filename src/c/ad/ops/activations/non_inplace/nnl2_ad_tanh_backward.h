#ifndef NNL2_AD_TANH_BACKWARD_H
#define NNL2_AD_TANH_BACKWARD_H

// NNL2

/** @file nnl2_ad_tanh_backward.h
 ** @brief Reverse mode derivative implementation for tanh operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Compute derivative of tanh operation and propagate gradients
 *
 ** @param ad_tensor
 * Output tensor from tanh forward pass
 *
 ** @param tensor_root
 * Input tensor to tanh operation
 *
 ** @param approx
 * Whether to use approximate tanh calculation (currently unused)
 *
 ** @details
 * Derivative of tanh(x) is: 1 - tanh²(x)
 * Note: approx parameter is currently unused but kept for API consistency
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
 ** @see nnl2_ad_tanh()
 ** @see nnl2_ad_reverse_backward_tanh()
 **/
void nnl2_ad_reverse_derivative_tanh(nnl2_ad_tensor* ad_tensor, nnl2_ad_tensor* tensor_root, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Mark unused parameter to avoid compiler warnings
	(void)approx;
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_tanh, ad_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor_root, "In function nnl2_ad_reverse_derivative_tanh, tensor_root is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_tanh, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor_root->data, "In function nnl2_ad_reverse_derivative_tanh, tensor_root data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->grad, "In function nnl2_ad_reverse_derivative_tanh, ad_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor_root->grad, "In function nnl2_ad_reverse_derivative_tanh, tensor_root grad is NULL");
	#endif
	
    nnl2_tensor* output = ad_tensor->data;
    nnl2_tensor* grad_output = ad_tensor->grad;
    nnl2_tensor* grad_input = tensor_root->grad;

    size_t total_elems = product(output->shape, output->rank);

    switch (output->dtype) {
        case FLOAT64: {
            double* y = (double*)output->data;
            double* grad_out = (double*)grad_output->data;
            double* grad_in = (double*)grad_input->data;
		
            for(size_t i = 0; i < total_elems; i++) {
                double t = y[i];
                grad_in[i] += grad_out[i] * (1.0 - t * t);
            }
			
            break;
        }

        case FLOAT32: {
            float* y = (float*)output->data;
            float* grad_out = (float*)grad_output->data;
            float* grad_in = (float*)grad_input->data;
			
            for(size_t i = 0; i < total_elems; i++) {
                float t = y[i];
                grad_in[i] += grad_out[i] * (1.0f - t * t);
            }
			
            break;
        }

        case INT32: {
            double* y = (double*)output->data;
            double* grad_out = (double*)grad_output->data;
            double* grad_in = (double*)grad_input->data;
			
            for(size_t i = 0; i < total_elems; i++) {
                double t = y[i];
                grad_in[i] += grad_out[i] * (1.0 - t * t);
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(output->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_TANH_BACKWARD_H **/
