#ifndef NNL2_AD_ATAN_BACKWARD_H
#define NNL2_AD_ATAN_BACKWARD_H

/** @file nnl2_ad_atan_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for arctangent operation
 **/

/** @brief 
 * Computes the gradient of the arctangent operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the arctangent operation
 *
 ** @param ad_tensor 
 * The input tensor to the arctangent operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Derivative formula: 1 / (1 + x^2)
 *
 ** @see nnl2_atan
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_atan(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!ad_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_atan, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_atan, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_atan, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_atan, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan, ad_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_atan, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->grad, "In function nnl2_ad_reverse_derivative_atan, ad_tensor grad is NULL");
	#endif
	
	// d(atan(x))/dx = 1 / (1 + x²)
	// dL/dx = dL/dy * (1 / (1 + x²))
	
	size_t numel = product(ad_tensor->data->shape, ad_tensor->data->rank);
    nnl2_tensor_type dtype = ad_tensor->data->dtype;

    switch(dtype) {
		case FLOAT64: {
			// Type-cast
			nnl2_float64* xd = (nnl2_float64*)ad_tensor->data->data;
			nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
			nnl2_float64* xg = (nnl2_float64*)ad_tensor->grad->data;

			for(size_t i = 0; i < numel; i++) {
				// x^2
				nnl2_float64 x_squared = xd[i] * xd[i];
				// 1 + x^2
				nnl2_float64 one_plus_x_squared = 1.0 + x_squared;
				// 1 / (1 + x^2)
				nnl2_float64 derivative = 1.0 / one_plus_x_squared;
				// dL/dx += dL/dy * derivative
				xg[i] += gd[i] * derivative;
			}
			
			break;
		}

		case FLOAT32: {
			// Type-cast
			nnl2_float32* xd = (nnl2_float32*)ad_tensor->data->data;
			nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
			nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;

			for(size_t i = 0; i < numel; i++) {
				// x^2
				nnl2_float32 x_squared = xd[i] * xd[i];
				// 1 + x^2
				nnl2_float32 one_plus_x_squared = 1.0f + x_squared;
				// 1 / (1 + x^2)
				nnl2_float32 derivative = 1.0f / one_plus_x_squared;
				// dL/dx += dL/dy * derivative
				xg[i] += gd[i] * derivative;
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

#endif /** NNL2_AD_ATAN_BACKWARD_H **/
