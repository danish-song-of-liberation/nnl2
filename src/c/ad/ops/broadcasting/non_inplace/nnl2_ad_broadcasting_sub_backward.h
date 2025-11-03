#ifndef NNL2_AD_SUB_BROADCASTING_BACKWARD_H
#define NNL2_AD_SUB_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_sub_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_min = minuend->grad;
    nnl2_tensor* grad_sub = subtrahend->grad;
    nnl2_tensor* data_sub = subtrahend->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_sub = product(data_sub->shape, data_sub->rank);
    size_t blocks = numel_out / numel_sub;

    if (minuend->requires_grad) nnl2_add_inplace(grad_min, grad_out);

    if (subtrahend->requires_grad) {
        switch (data_sub->dtype) {
            case FLOAT64: {
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gsub = (nnl2_float64*)grad_sub->data;

                for (size_t i = 0; i < numel_sub; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            case FLOAT32: {
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gsub = (nnl2_float32*)grad_sub->data;

                for (size_t i = 0; i < numel_sub; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            case INT32: {
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gsub = (nnl2_int32*)grad_sub->data;

                for (size_t i = 0; i < numel_sub; i++) {
                    nnl2_int32 acc = 0;
                    for (size_t b = 0; b < blocks; b++) acc += gout[b * numel_sub + i];
                    gsub[i] -= acc;
                }

                break;
            }

            default: NNL2_TYPE_ERROR(data_sub->dtype);
        }
    }
}

#endif /** NNL2_AD_SUB_BROADCASTING_BACKWARD_H **/
