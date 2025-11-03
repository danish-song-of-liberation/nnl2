#ifndef NNL2_AD_MAX_BROADCASTING_BACKWARD_H
#define NNL2_AD_MAX_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_max_broadcasting(
    nnl2_ad_tensor* out_tensor,
    nnl2_ad_tensor* a,
    nnl2_ad_tensor* b)
{
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* data_a = a->data;
    nnl2_tensor* data_b = b->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_a = product(data_a->shape, data_a->rank);
    size_t numel_b = product(data_b->shape, data_b->rank);

    size_t blocks_a = numel_out / numel_a;
    size_t blocks_b = numel_out / numel_b;

    switch (data_a->dtype) {
        case FLOAT64: {
            nnl2_float64* ad = (nnl2_float64*)data_a->data;
            nnl2_float64* bd = (nnl2_float64*)data_b->data;
            nnl2_float64* gd = (nnl2_float64*)grad_out->data;

            if (a->requires_grad) {
                nnl2_float64* ag = (nnl2_float64*)a->grad->data;
                for (size_t i = 0; i < numel_a; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if (ad[i] > bd[j % numel_b]) acc += gd[j];
                    }
                    ag[i] += acc;
                }
            }

            if (b->requires_grad) {
                nnl2_float64* bg = (nnl2_float64*)b->grad->data;
                for (size_t i = 0; i < numel_b; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if (bd[i] > ad[j % numel_a]) acc += gd[j];
                    }
                    bg[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
            nnl2_float32* ad = (nnl2_float32*)data_a->data;
            nnl2_float32* bd = (nnl2_float32*)data_b->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;

            if (a->requires_grad) {
                nnl2_float32* ag = (nnl2_float32*)a->grad->data;
                for (size_t i = 0; i < numel_a; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if (ad[i] > bd[j % numel_b]) acc += gd[j];
                    }
                    ag[i] += acc;
                }
            }

            if (b->requires_grad) {
                nnl2_float32* bg = (nnl2_float32*)b->grad->data;
                for (size_t i = 0; i < numel_b; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if (bd[i] > ad[j % numel_a]) acc += gd[j];
                    }
                    bg[i] += acc;
                }
            }

            break;
        }

        case INT32: {
            nnl2_int32* ad = (nnl2_int32*)data_a->data;
            nnl2_int32* bd = (nnl2_int32*)data_b->data;
            nnl2_float32* gd = (nnl2_float32*)grad_out->data;

            if (a->requires_grad) {
                nnl2_float32* ag = (nnl2_float32*)a->grad->data;
                for (size_t i = 0; i < numel_a; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t bidx = 0; bidx < blocks_a; bidx++) {
                        size_t j = bidx * numel_a + i;
                        if (ad[i] > bd[j % numel_b]) acc += gd[j];
                    }
                    ag[i] += acc;
                }
            }

            if (b->requires_grad) {
                nnl2_float32* bg = (nnl2_float32*)b->grad->data;
                for (size_t i = 0; i < numel_b; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t bidx = 0; bidx < blocks_b; bidx++) {
                        size_t j = bidx * numel_b + i;
                        if (bd[i] > ad[j % numel_a]) acc += gd[j];
                    }
                    bg[i] += acc;
                }
            }

            break;
        }

        default:
            NNL2_TYPE_ERROR(data_a->dtype);
    }
}

#endif /** NNL2_AD_MAX_BROADCASTING_BACKWARD_H **/
