#ifndef NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H
#define NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H

void nnl2_ad_reverse_derivative_min_correspondence(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* threshold_tensor, void* threshold) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_in  = threshold_tensor->grad;
    nnl2_tensor* data_in  = threshold_tensor->data;

    switch (data_in->dtype) {
        case FLOAT64: {
            nnl2_float64 th = *((nnl2_float64*)threshold);
            nnl2_float64* x = (nnl2_float64*)data_in->data;
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;
            nnl2_float64* gin  = (nnl2_float64*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0.0;
            break;
        }

        case FLOAT32: {
            nnl2_float32 th = *((nnl2_float32*)threshold);
            nnl2_float32* x = (nnl2_float32*)data_in->data;
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;
            nnl2_float32* gin  = (nnl2_float32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0.0f;
            break;
        }

        case INT32: {
            nnl2_int32 th = *((nnl2_int32*)threshold);
            nnl2_int32* x = (nnl2_int32*)data_in->data;
            nnl2_int32* gout = (nnl2_int32*)grad_out->data;
            nnl2_int32* gin  = (nnl2_int32*)grad_in->data;
            size_t n = product(data_in->shape, data_in->rank);
            for(size_t i = 0; i < n; i++) gin[i] += (x[i] < th) ? gout[i] : 0;
            break;
        }

        default: NNL2_TYPE_ERROR(data_in->dtype);
    }
}

#endif /** NNL2_AD_CORRESPONDENCE_MIN_BACKWARD_H **/
