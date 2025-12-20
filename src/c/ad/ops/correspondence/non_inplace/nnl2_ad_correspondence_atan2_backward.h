#ifndef NNL2_AD_CORRESPONDENCE_ATAN2_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_ATAN2_BACKWARD_H

/** @file nnl2_ad_correspondence_atan2_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for correspondence atan2 operation
 **/

/** @brief 
 * Computes the gradient of the correspondence atan2 operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the correspondence atan2 operation
 *
 ** @param y_tensor 
 * The y-coordinate input tensor to the correspondence atan2 operation
 *
 ** @param threshold
 * The threshold value pointer (x-coordinate)
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Computes gradient as: d/dy = x / (x^2 + y^2) where x is the threshold
 *
 ** @see product
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_atan2_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* y_tensor, void* threshold) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    if(!y_tensor->requires_grad) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Exiting nnl2_ad_reverse_derivative_atan2_correspondence because y_tensor is not requiring gradient");
			NNL2_FUNC_EXIT();
		#endif
		
		return;
	}
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_atan2_correspondence, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y_tensor, "In function nnl2_ad_reverse_derivative_atan2_correspondence, y_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(threshold, "In function nnl2_ad_reverse_derivative_atan2_correspondence, threshold is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_atan2_correspondence, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y_tensor->data, "In function nnl2_ad_reverse_derivative_atan2_correspondence, y_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan2_correspondence, out_tensor data shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(y_tensor->data->shape, "In function nnl2_ad_reverse_derivative_atan2_correspondence, y_tensor data shape is NULL");
	#endif
    
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = y_tensor->grad;
    nnl2_tensor* data_in  = y_tensor->data;
    
    size_t n = product(data_in->shape, data_in->rank);

    switch(data_in->dtype) {
        case FLOAT64: {
            nnl2_float64 x_val = *((nnl2_float64*)threshold);
            nnl2_float64* y = (nnl2_float64*)data_in->data;
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gin  = (nnl2_float64*)grad_in->data;
            
            for(size_t i = 0; i < n; i++) {
                nnl2_float64 y_val = y[i];
                nnl2_float64 denominator = x_val * x_val + y_val * y_val;
                
                if(denominator != 0.0) {
                    gin[i] += x_val * gout[i] / denominator;
                }
            }
			
            break;
        }

        case FLOAT32: {
			// Type-cast
            nnl2_float32 x_val = *((nnl2_float32*)threshold);
            nnl2_float32* y = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            
            for(size_t i = 0; i < n; i++) {
                nnl2_float32 y_val = y[i];
                nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                
                if(denominator != 0.0f) {
                    gin[i] += x_val * gout[i] / denominator;
                }
            }
			
            break;
        }

        case INT32: {
            nnl2_float32 x_val = (nnl2_float32)(*((nnl2_int32*)threshold));
            nnl2_float32* y = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            
            for(size_t i = 0; i < n; i++) {
                nnl2_float32 y_val = y[i];
                nnl2_float32 denominator = x_val * x_val + y_val * y_val;
                
                if(denominator != 0.0f) {
                    gin[i] += x_val * gout[i] / denominator;
                }
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(data_in->dtype);
            break;
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CORRESPONDENCE_ATAN2_BACKWARD_H **/
