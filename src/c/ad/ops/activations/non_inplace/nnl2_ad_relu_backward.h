#ifndef NNL2_AD_RELU_BACKWARD_H
#define NNL2_AD_RELU_BACKWARD_H

void nnl2_ad_reverse_derivative_relu(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a) {
    size_t numel = product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)a->data->data;          
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data; 
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;          
            for (size_t i = 0; i < numel; i++) ag[i] += (ad[i] > 0.0) ? gd[i] : 0.0;
            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) ag[i] += (ad[i] > 0.0f) ? gd[i] : 0.0f;
            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;
            for (size_t i = 0; i < numel; i++) ag[i] += (ad[i] > 0) ? gd[i] : 0.0f;
            break;
        }

        default: {
            NNL2_TYPE_ERROR(dtype);
            break;
        }
    }
}


#endif /** NNL2_AD_RELU_BACKWARD_H **/
