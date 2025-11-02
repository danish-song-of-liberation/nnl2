#ifndef NNL2_AD_SIGMOID_BACKWARD_H
#define NNL2_AD_SIGMOID_BACKWARD_H

void nnl2_ad_reverse_derivative_sigmoid(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* a, bool approx) {
    size_t numel = product(a->data->shape, a->data->rank);
    nnl2_tensor_type dtype = a->data->dtype;

    switch (dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)a->data->data;
            nnl2_float64* gd = (nnl2_float64*)out_tensor->grad->data;
            nnl2_float64* ag = (nnl2_float64*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 s;
                if(approx) {
                    nnl2_float64 x = ad[i];
                    nnl2_float64 abs_x = fabs(x);
                    s = 0.5 + 0.5 * x / (1.0 + abs_x);
                } else {
                    s = 1.0 / (1.0 + exp(-ad[i]));
                }
                ag[i] += gd[i] * s * (1.0 - s);
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float32 s;
                if(approx) {
                    nnl2_float32 x = ad[i];
                    if (x < -8.0f) s = 0.0f;
                    else if (x > 8.0f) s = 1.0f;
                    else {
                        nnl2_float32 abs_x = fabsf(x);
                        s = 0.5f + 0.5f * x / (1.0f + abs_x);
                    }
                } else {
                    s = 1.0f / (1.0f + expf(-ad[i]));
                }
                ag[i] += gd[i] * s * (1.0f - s);
            }
			
            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)a->data->data;
            nnl2_float32* gd = (nnl2_float32*)out_tensor->grad->data;
            nnl2_float32* ag = (nnl2_float32*)a->grad->data;

            for(size_t i = 0; i < numel; i++) {
                nnl2_float64 x = (nnl2_float64)ad[i];
                nnl2_float64 s;
                if(approx) {
                    nnl2_float64 abs_x = fabs(x);
                    s = 0.5 + 0.5 * x / (1.0 + abs_x);
                } else {
                    s = 1.0 / (1.0 + exp(-x));
                }
                ag[i] += gd[i] * (nnl2_float32)(s * (1.0 - s));
            }
			
            break;
        }

        default:
            NNL2_TYPE_ERROR(dtype);
            break;
    }
}

#endif /** NNL2_AD_SIGMOID_BACKWARD_H **/
