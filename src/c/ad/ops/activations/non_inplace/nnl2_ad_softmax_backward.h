#ifndef NNL2_AD_SOFTMAX_BACKWARD_H
#define NNL2_AD_SOFTMAX_BACKWARD_H

// NNL2

/** @file nnl2_ad_softmax_backward.h
 ** @brief Reverse mode derivative implementation for Softmax operation
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Compute derivative of Softmax operation and propagate gradients
 *
 ** @param out_tensor
 * Output tensor from Softmax forward pass
 *
 ** @param a
 * Input tensor to Softmax operation
 *
 ** @param dim
 * Dimension along which softmax was computed
 *
 ** @details
 * Derivative of Softmax dS_i/dx_j = S_i(delta_ij - S_j)
 * For backward pass: dL/dx_i = S_i * (dL/dS_i - Σ_j S_j * dL/dS_j)
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
 ** @see nnl2_ad_softmax()
 ** @see nnl2_ad_reverse_backward_softmax()
 **/
void nnl2_ad_reverse_derivative_softmax(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, int dim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_softmax, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a, "In function nnl2_ad_reverse_derivative_softmax, input tensor a is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_softmax, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->data, "In function nnl2_ad_reverse_derivative_softmax, input tensor a data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_softmax, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(a->grad, "In function nnl2_ad_reverse_derivative_softmax, input tensor a grad is NULL");
	#endif
	
	// Dimension validation
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (dim < 0 || dim >= a->data->rank) {
			NNL2_ERROR("In function nnl2_ad_reverse_derivative_softmax, invalid dimension %d for tensor of rank %d", dim, a->data->rank);
			return;
		}
	#endif
	
	int32_t dim_size = a->data->shape[dim];
	int32_t stride_dim = a->data->strides[dim];
	size_t total_elements = nnl2_product(a->data->shape, a->data->rank);
	nnl2_tensor_type dtype = a->data->dtype;

	switch (dtype) {
		case FLOAT32: {
			nnl2_float32* s = (nnl2_float32*)out_tensor->data->data;   
			nnl2_float32* g = (nnl2_float32*)out_tensor->grad->data;  //dL/dS
			nnl2_float32* a_grad = (nnl2_float32*)a->grad->data; // dL/dx 
			
			for (size_t idx = 0; idx < total_elements; idx += dim_size * stride_dim) { 
				nnl2_float32 sum = 0.0f;
				for (int32_t j = 0; j < dim_size; j++) {
					size_t j_idx = idx + j * stride_dim;
					sum += s[j_idx] * g[j_idx];
				}
				
				// dL/dx_i = S_i * (dL/dS_i - sum)
				for (int32_t i = 0; i < dim_size; i++) {
					size_t i_idx = idx + i * stride_dim;
					a_grad[i_idx] += s[i_idx] * (g[i_idx] - sum);
				}
			}
			break;
		}
		
		case FLOAT64: {
			nnl2_float64* s = (nnl2_float64*)out_tensor->data->data;        
			nnl2_float64* g = (nnl2_float64*)out_tensor->grad->data; // dL/∂S
			nnl2_float64* a_grad = (nnl2_float64*)a->grad->data; //dL/dx 
			
			for(size_t idx = 0; idx < total_elements; idx += dim_size * stride_dim) {
				nnl2_float64 sum = 0.0;
				for(int32_t j = 0; j < dim_size; j++) {
					size_t j_idx = idx + j * stride_dim;
					sum += s[j_idx] * g[j_idx];
				}
				
				for(int32_t i = 0; i < dim_size; i++) {
					size_t i_idx = idx + i * stride_dim;
					a_grad[i_idx] += s[i_idx] * (g[i_idx] - sum);
				}
			}
			break;
		}
		
		case INT32: {
			nnl2_float32* s = (nnl2_float32*)out_tensor->data->data;          
			nnl2_float32* g = (nnl2_float32*)out_tensor->grad->data; // dL/dS
			nnl2_float32* a_grad = (nnl2_float32*)a->grad->data; // dL/dx 
			
			for(size_t idx = 0; idx < total_elements; idx += dim_size * stride_dim) {
				nnl2_float32 sum = 0.0f;
				for (int32_t j = 0; j < dim_size; j++) {
					size_t j_idx = idx + j * stride_dim;
					sum += s[j_idx] * g[j_idx];
				}
				
				// dL/dx_i = S_i * (dL/dS_i - sum)
				for (int32_t i = 0; i < dim_size; i++) {
					size_t i_idx = idx + i * stride_dim;
					a_grad[i_idx] += s[i_idx] * (g[i_idx] - sum);
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

#endif /** NNL2_AD_SOFTMAX_BACKWARD_H **/
