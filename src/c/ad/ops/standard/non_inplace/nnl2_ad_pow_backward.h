#ifndef NNL2_AD_POW_BACKWARD_H
#define NNL2_AD_POW_BACKWARD_H

void nnl2_ad_reverse_derivative_pow(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* base, nnl2_ad_tensor* exponent) {
	//todo fix crutch
	
    size_t numel = product(base->data->shape, base->data->rank);
    nnl2_tensor_type dtype = base->data->dtype;
	
	switch(dtype) {
        case FLOAT64: {
            nnl2_float64* a_data = (nnl2_float64*)base->data->data;
            nnl2_float64* b_data = (nnl2_float64*)exponent->data->data;
            nnl2_float64* out_grad = (nnl2_float64*)out_tensor->grad->data;

            if(base->requires_grad) {
                nnl2_float64* a_grad = (nnl2_float64*)base->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0.0) a_grad[i] += out_grad[i] * b_data[i] * pow(a_data[i], b_data[i] - 1.0);
                }
            }

            if(exponent->requires_grad) {
                nnl2_float64* b_grad = (nnl2_float64*)exponent->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if (a_data[i] > 0.0) b_grad[i] += out_grad[i] * pow(a_data[i], b_data[i]) * log(a_data[i]);
                }
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* a_data = (nnl2_float32*)base->data->data;
            nnl2_float32* b_data = (nnl2_float32*)exponent->data->data;
            nnl2_float32* out_grad = (nnl2_float32*)out_tensor->grad->data;

            if(base->requires_grad) {
                nnl2_float32* a_grad = (nnl2_float32*)base->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0.0f) a_grad[i] += out_grad[i] * b_data[i] * powf(a_data[i], b_data[i] - 1.0f);
                }
            }

            if(exponent->requires_grad) {
                nnl2_float32* b_grad = (nnl2_float32*)exponent->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] > 0.0f) b_grad[i] += out_grad[i] * powf(a_data[i], b_data[i]) * logf(a_data[i]);
                }
            }
			
            break;
        }

        case INT32: {
            nnl2_int32* a_data = (nnl2_int32*)base->data->data;
            nnl2_int32* b_data = (nnl2_int32*)exponent->data->data;
            nnl2_float32* out_grad = (nnl2_float32*)out_tensor->grad->data; 

            if(base->requires_grad) {
                nnl2_float32* a_grad = (nnl2_float32*)base->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] != 0) a_grad[i] += out_grad[i] * b_data[i] * powf((float)a_data[i], (float)b_data[i] - 1.0f);
                }
            }

            if(exponent->requires_grad) {
                nnl2_float32* b_grad = (nnl2_float32*)exponent->grad->data;
                for(size_t i = 0; i < numel; i++) {
                    if(a_data[i] > 0) b_grad[i] += out_grad[i] * powf((float)a_data[i], (float)b_data[i]) * logf((float)a_data[i]);
                }
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(dtype);
            break;
        }
    }
}

#endif /** NNL2_AD_POW_BACKWARD_H **/
