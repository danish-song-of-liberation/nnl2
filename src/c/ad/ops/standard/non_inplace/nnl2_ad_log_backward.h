#ifndef NNL2_AD_LOG_BACKWARD_DERIVATIVE_H
#define NNL2_AD_LOG_BACKWARD_DERIVATIVE_H

void nnl2_ad_reverse_derivative_log(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
    size_t numel = product(ad_tensor->data->shape, ad_tensor->data->rank);
    nnl2_tensor_type dtype = ad_tensor->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* xd = (nnl2_float64*)ad_tensor->data->data;
            nnl2_float64* yg = (nnl2_float64*)out_tensor->grad->data;

            if (ad_tensor->requires_grad) {
                nnl2_float64* xg = (nnl2_float64*)ad_tensor->grad->data;
                for (size_t i = 0; i < numel; i++) {
                    xg[i] += yg[i] / xd[i];
                }
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* xd = (nnl2_float32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;

            if (ad_tensor->requires_grad) {
                nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
                for (size_t i = 0; i < numel; i++) {
                    xg[i] += yg[i] / xd[i];
                }
            }
			
            break;
        }

        case INT32: {
            nnl2_int32* xd = (nnl2_int32*)ad_tensor->data->data;
            nnl2_float32* yg = (nnl2_float32*)out_tensor->grad->data;

            if (ad_tensor->requires_grad) {
                nnl2_float32* xg = (nnl2_float32*)ad_tensor->grad->data;
                for (size_t i = 0; i < numel; i++) {
                    if (xd[i] != 0) xg[i] += yg[i] / (nnl2_float32)xd[i];
                }
            }
			
            break;
        }

        default:
            NNL2_TYPE_ERROR(dtype);
    }
}

#endif /** NNL2_AD_LOG_BACKWARD_DERIVATIVE_H **/
