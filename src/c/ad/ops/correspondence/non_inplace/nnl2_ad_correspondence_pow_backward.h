#ifndef NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H

void nnl2_ad_reverse_derivative_pow_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* exponent_tensor, void* exponent) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = exponent_tensor->grad;
    nnl2_tensor* data_in  = exponent_tensor->data;

    switch (data_in->dtype) {
        case FLOAT64: {
            nnl2_float64 k = *((nnl2_float64*)exponent);
            nnl2_float64* x = (nnl2_float64*)data_in->data;
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gin  = (nnl2_float64*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += gout[i] * k * pow(x[i], k - 1.0);
            break;
        }

        case FLOAT32: {
            nnl2_float32 k = *((nnl2_float32*)exponent);
            nnl2_float32* x = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += gout[i] * k * powf(x[i], k - 1.0f);
            break;
        }

        case INT32: {
            nnl2_int32 k = *((nnl2_int32*)exponent);
            nnl2_int32* x = (nnl2_int32*)data_in->data;
            nnl2_int32* gout = (nnl2_int32*)grad_out->data;
            nnl2_int32* gin  = (nnl2_int32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
			
            for(size_t i = 0; i < n; i++) {
                nnl2_int32 p = 1;
                for (int j = 0; j < k - 1; j++) p *= x[i];
                gin[i] += gout[i] * k * p;
            }
			
            break;
        }

        default: NNL2_TYPE_ERROR(data_in->dtype);
    }
}

#endif /** NNL2_AD_CORRESPONDENCE_POW_BACKWARD_H **/
