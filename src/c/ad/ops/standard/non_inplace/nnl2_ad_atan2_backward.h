#ifndef NNL2_AD_ATAN2_BACKWARD_DERIVATIVE_H
#define NNL2_AD_ATAN2_BACKWARD_DERIVATIVE_H

/** @file nnl2_ad_atan2_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for atan2 operation
 **/

/** @brief 
 * Computes the gradient of the atan2 operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the atan2 operation
 *
 ** @param y 
 * The first input tensor to the atan2 operation (y-coordinate)
 *
 ** @param x 
 * The second input tensor to the atan2 operation (x-coordinate)
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Computes gradients as:
 * d/dy = x / (x^2 + y^2)
 * d/dx = -y / (x^2 + y^2)
 *
 ** @see nnl2_product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_atan2(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* y, nnl2_ad_tensor* x) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_atan2, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y, "In function nnl2_ad_reverse_derivative_atan2, y is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x, "In function nnl2_ad_reverse_derivative_atan2, x is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_atan2, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y->data, "In function nnl2_ad_reverse_derivative_atan2, y data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x->data, "In function nnl2_ad_reverse_derivative_atan2, x data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan2, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y->data->shape, "In function nnl2_ad_reverse_derivative_atan2, y data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x->data->shape, "In function nnl2_ad_reverse_derivative_atan2, x data shape is NULL");
	#endif
    
    size_t numel = nnl2_product(y->data->shape, y->data->rank);
    nnl2_tensor_type dtype = y->data->dtype;

    switch(dtype) {
		case FLOAT64: {
			// Type-cast
			nnl2_float64* yd = (nnl2_float64*)y->data->data;
			nnl2_float64* xd = (nnl2_float64*)x->data->data;
			nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
			nnl2_float64* yg = (nnl2_float64*)y->grad->data;
			nnl2_float64* xg = (nnl2_float64*)x->grad->data;

			if(y->requires_grad || x->requires_grad) {
				for(size_t i = 0; i < numel; i++) {
					nnl2_float64 y_val = yd[i];
					nnl2_float64 x_val = xd[i];
					nnl2_float64 denominator = x_val * x_val + y_val * y_val;

					if(denominator == 0.0) {
						if(y->requires_grad) yg[i] += 0.0;
						if(x->requires_grad) xg[i] += 0.0;
					} else {
						nnl2_float64 grad_scale = gd[i] / denominator;
						
						if(y->requires_grad) {
							yg[i] += x_val * grad_scale;  // d/dy = x / (x^2 + y^2)
						}
						
						if(x->requires_grad) {
							xg[i] += -y_val * grad_scale; // d/dx = -y / (x^2 + y^2)
						}
					}
				}
			}
			break;
		}

		case FLOAT32: {
			// Type-cast
			nnl2_float32* yd = (nnl2_float32*)y->data->data;
			nnl2_float32* xd = (nnl2_float32*)x->data->data;
			nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
			nnl2_float32* yg = (nnl2_float32*)y->grad->data;
			nnl2_float32* xg = (nnl2_float32*)x->grad->data;

			if(y->requires_grad || x->requires_grad) {
				for(size_t i = 0; i < numel; i++) {
					nnl2_float32 y_val = yd[i];
					nnl2_float32 x_val = xd[i];
					nnl2_float32 denominator = x_val * x_val + y_val * y_val;
					
					// Avoid division by zero
					if(denominator == 0.0f) {
						if(y->requires_grad) yg[i] += 0.0f;
						if(x->requires_grad) xg[i] += 0.0f;
					} else {
						nnl2_float32 grad_scale = gd[i] / denominator;
						
						if(y->requires_grad) {
							yg[i] += x_val * grad_scale;  // d/dy = x / (x^2 + y^2)
						}
						
						if(x->requires_grad) {
							xg[i] += -y_val * grad_scale; // d/dx = -y / (x^2 + y^2)
						}
					}
				}
			}
			break;
		}

		case INT32: {
			nnl2_int32* yd = (nnl2_int32*)y->data->data;
			nnl2_int32* xd = (nnl2_int32*)x->data->data;
			nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
			nnl2_float32* yg = (nnl2_float32*)y->grad->data;
			nnl2_float32* xg = (nnl2_float32*)x->grad->data;

			if(y->requires_grad || x->requires_grad) {
				for(size_t i = 0; i < numel; i++) {
					nnl2_float32 y_val = (nnl2_float32)yd[i];
					nnl2_float32 x_val = (nnl2_float32)xd[i];
					nnl2_float32 denominator = x_val * x_val + y_val * y_val;
					
					// Avoid division by zero
					if(denominator == 0.0f) {
						if(y->requires_grad) yg[i] += 0.0f;
						if(x->requires_grad) xg[i] += 0.0f;
					} else {
						nnl2_float32 grad_scale = gd[i] / denominator;
						
						if(y->requires_grad) {
							yg[i] += x_val * grad_scale;  // d/dy = x / (x^2 + y^2)
						}
						
						if(x->requires_grad) {
							xg[i] += -y_val * grad_scale; // d/dx = -y / (x^2 + y^2)
						}
					}
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

#endif /** NNL2_AD_ATAN2_BACKWARD_DERIVATIVE_H **/
