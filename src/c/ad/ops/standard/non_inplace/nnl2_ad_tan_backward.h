#ifndef NNL2_AD_TAN_BACKWARD_H
#define NNL2_AD_TAN_BACKWARD_H

/** @file nnl2_ad_tan_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for tangent operation
 **/

/** @brief 
 * Computes the gradient of the tangent operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the tangent operation
 *
 ** @param ad_tensor 
 * The input tensor to the tangent operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Derivative formula: sec^2(x) = 1 + tan^2(x)
 *
 ** @see nnl2_tan
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_tan(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
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
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_tan, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_reverse_derivative_tan, ad_tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_tan, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data, "In function nnl2_ad_reverse_derivative_tan, ad_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_tan, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->data->shape, "In function nnl2_ad_reverse_derivative_tan, ad_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_tan, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor->grad, "In function nnl2_ad_reverse_derivative_tan, ad_tensor grad is NULL");
	#endif
	
	// d(tan(x))/dx = sec^2(x) = 1 + tan^2(x)
	// dL/dx = dL/dy * (1 + tan^2(x))
	
	size_t numel = nnl2_product(ad_tensor->data->shape, ad_tensor->data->rank);
    nnl2_tensor_type dtype = ad_tensor->data->dtype;

    switch(dtype) {
		case FLOAT64: {
			// Type-cast
			nnl2_float64* xd = (nnl2_float64*)ad_tensor->data->data;
			nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
			nnl2_float64* xg = (nnl2_float64*)ad_tensor->grad->data;

			// Process each element
			for(size_t i = 0; i < numel; i++) {
				// tan(x) 
				nnl2_float64 tan_x = tan(xd[i]);
				// 1 + tan^2(x)
				nnl2_float64 derivative = 1.0 + (tan_x * tan_x);
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
				// tan(x) 
				nnl2_float32 tan_x = tanf(xd[i]);
				// 1 + tan^2(x)
				nnl2_float32 derivative = 1.0f + (tan_x * tan_x);
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

#endif /** NNL2_AD_TAN_BACKWARD_H **/
