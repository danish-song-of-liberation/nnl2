#ifndef NNL2_AD_ABS_BACKWARD_H
#define NNL2_AD_ABS_BACKWARD_H

void nnl2_ad_reverse_derivative_abs(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
    if(!ad_tensor->requires_grad) return;
    
    size_t numel = product(out_tensor->data->shape, out_tensor->data->rank);
    
    switch(ad_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64* out_grad_data = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* input_data = (nnl2_float64*)ad_tensor->data->data;
            nnl2_float64* input_grad_data = (nnl2_float64*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 sign;
                
                if(input_data[i] > 0) {
                    sign = 1.0;
                } else if (input_data[i] < 0) {
                    sign = -1.0;
                } else {
                    sign = 0.0;
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* out_grad_data = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* input_data = (nnl2_float32*)ad_tensor->data->data;
            nnl2_float32* input_grad_data = (nnl2_float32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 sign;
                
                if(input_data[i] > 0) {
                    sign = 1.0f;
                } else if (input_data[i] < 0) {
                    sign = -1.0f;
                } else {
                    sign = 0.0f;
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        case INT32: {
            nnl2_int32* out_grad_data = (nnl2_int32*)out_tensor->grad->data;
            nnl2_int32* input_data = (nnl2_int32*)ad_tensor->data->data;
            nnl2_int32* input_grad_data = (nnl2_int32*)ad_tensor->grad->data;
            
            for(size_t i = 0; i < numel; i++) {
                nnl2_int32 sign;
                
                if(input_data[i] > 0) {
                    sign = 1;
                } else if (input_data[i] < 0) {
                    sign = -1;
                } else {
                    sign = 0;
                }
                
                input_grad_data[i] += out_grad_data[i] * sign;
            }
			
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(ad_tensor->data->dtype);
            break;
        }
    }
}

#endif /** NNL2_AD_ABS_BACKWARD_H **/
