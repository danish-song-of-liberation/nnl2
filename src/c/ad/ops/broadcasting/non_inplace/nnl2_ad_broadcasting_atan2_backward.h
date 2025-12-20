#ifndef NNL2_AD_ATAN2_BROADCASTING_BACKWARD_H
#define NNL2_AD_ATAN2_BROADCASTING_BACKWARD_H

/** @file nnl2_ad_broadcasting_atan2_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for broadcasting atan2 operation
 **/

/** @brief 
 * Computes the gradient of the broadcasting atan2 operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the broadcasting atan2 operation
 *
 ** @param y 
 * The first input tensor to the broadcasting atan2 operation (y-coordinate)
 *
 ** @param x 
 * The second input tensor to the broadcasting atan2 operation (x-coordinate)
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Computes gradients with broadcasting as:
 * d/dy = x / (x^2 + y^2)
 * d/dx = -y / (x^2 + y^2)
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_atan2_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* y, nnl2_ad_tensor* x) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, y is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, x is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y->data, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, y data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x->data, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, x data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y->data->shape, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, y data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(x->data->shape, "In function nnl2_ad_reverse_derivative_atan2_broadcasting, x data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* data_y = y->data;
    nnl2_tensor* data_x = x->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_y = product(data_y->shape, data_y->rank);
    size_t numel_x = product(data_x->shape, data_x->rank);

    size_t blocks_y = numel_out / numel_y;
    size_t blocks_x = numel_out / numel_x;

    switch(data_y->dtype) {
        case FLOAT64: {
			// Type-cast
            nnl2_float64* yd = (nnl2_float64*)data_y->data;
            nnl2_float64* xd = (nnl2_float64*)data_x->data;
            nnl2_float64* gd = (nnl2_float64*)grad_out->data;
            nnl2_float64* yg = (nnl2_float64*)y->grad->data;
            nnl2_float64* xg = (nnl2_float64*)x->grad->data;

            if(y->requires_grad) {
                for(size_t i = 0; i < numel_y; i++) {
                    nnl2_float64 acc = 0.0;
					
                    for(size_t bidx = 0; bidx < blocks_y; bidx++) {
                        size_t j = bidx * numel_y + i;
                        size_t x_idx = j % numel_x;
                        nnl2_float64 y_val = yd[i];
                        nnl2_float64 x_val = xd[x_idx];
                        nnl2_float64 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0) {
                            acc += x_val * gd[j] / denominator;
                        }
                    }
					
                    yg[i] += acc;
                }
            }

            if(x->requires_grad) {
                for(size_t i = 0; i < numel_x; i++) {
                    nnl2_float64 acc = 0.0;
					
                    for(size_t bidx = 0; bidx < blocks_x; bidx++) {
                        size_t j = bidx * numel_x + i;
                        size_t y_idx = j % numel_y;
                        nnl2_float64 y_val = yd[y_idx];
                        nnl2_float64 x_val = xd[i];
                        nnl2_float64 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0) {
                            acc += -y_val * gd[j] / denominator;
                        }
                    }
					
                    xg[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
            nnl2_float32* yd = (nnl2_float32*)data_y->data;
            nnl2_float32* xd = (nnl2_float32*)data_x->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;
            nnl2_float32* yg = (nnl2_float32*)y->grad->data;
            nnl2_float32* xg = (nnl2_float32*)x->grad->data;

            if(y->requires_grad) {
                for(size_t i = 0; i < numel_y; i++) {
                    nnl2_float32 acc = 0.0f;
					
                    for(size_t bidx = 0; bidx < blocks_y; bidx++) {
                        size_t j = bidx * numel_y + i;
                        size_t x_idx = j % numel_x;
                        nnl2_float32 y_val = yd[i];
                        nnl2_float32 x_val = xd[x_idx];
                        nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0f) {
                            acc += x_val * gd[j] / denominator;
                        }
                    }
					
                    yg[i] += acc;
                }
            }

            if(x->requires_grad) {
                for(size_t i = 0; i < numel_x; i++) {
                    nnl2_float32 acc = 0.0f;
					
                    for(size_t bidx = 0; bidx < blocks_x; bidx++) {
                        size_t j = bidx * numel_x + i;
                        size_t y_idx = j % numel_y;
                        nnl2_float32 y_val = yd[y_idx];
                        nnl2_float32 x_val = xd[i];
                        nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0f) {
                            acc += -y_val * gd[j] / denominator;
                        }
                    }
					
                    xg[i] += acc;
                }
            }

            break;
        }

        case INT32: {
            nnl2_int32* yd = (nnl2_int32*)data_y->data;
            nnl2_int32* xd = (nnl2_int32*)data_x->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;
            nnl2_float32* yg = (nnl2_float32*)y->grad->data;
            nnl2_float32* xg = (nnl2_float32*)x->grad->data;

            if(y->requires_grad) {
                for(size_t i = 0; i < numel_y; i++) {
                    nnl2_float32 acc = 0.0f;
					
                    for(size_t bidx = 0; bidx < blocks_y; bidx++) {
                        size_t j = bidx * numel_y + i;
                        size_t x_idx = j % numel_x;
                        nnl2_float32 y_val = (nnl2_float32)yd[i];
                        nnl2_float32 x_val = (nnl2_float32)xd[x_idx];
                        nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0f) {
                            acc += x_val * gd[j] / denominator;
                        }
                    }
					
                    yg[i] += acc;
                }
            }

            if(x->requires_grad) {
                for(size_t i = 0; i < numel_x; i++) {
                    nnl2_float32 acc = 0.0f;
					
                    for(size_t bidx = 0; bidx < blocks_x; bidx++) {
                        size_t j = bidx * numel_x + i;
                        size_t y_idx = j % numel_y;
                        nnl2_float32 y_val = (nnl2_float32)yd[y_idx];
                        nnl2_float32 x_val = (nnl2_float32)xd[i];
                        nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                        
                        if(denominator != 0.0f) {
                            acc += -y_val * gd[j] / denominator;
                        }
                    }
					
                    xg[i] += acc;
                }
            }

            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_y->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_ATAN2_BROADCASTING_BACKWARD_H **/
