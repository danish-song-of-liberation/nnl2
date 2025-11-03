#ifndef NNL2_AD_MUL_BROADCASTING_BACKWARD_H
#define NNL2_AD_MUL_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_mul_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* multiplier, nnl2_ad_tensor* multiplicand) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_mult = multiplier->grad;
    nnl2_tensor* grad_cand = multiplicand->grad;
    nnl2_tensor* data_mult = multiplier->data;
    nnl2_tensor* data_cand = multiplicand->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_mult = product(data_mult->shape, data_mult->rank);
    size_t numel_cand = product(data_cand->shape, data_cand->rank);
    size_t blocks_for_mult = numel_out / numel_mult;
    size_t blocks_for_cand = numel_out / numel_cand;

    if (multiplier->requires_grad) {
        switch (data_mult->dtype) {
            case FLOAT64: {
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gmul = (nnl2_float64*)grad_mult->data;
                nnl2_float64* bcand = (nnl2_float64*)data_cand->data;

                for (size_t i = 0; i < numel_mult; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks_for_mult; b++) acc += gout[b * numel_mult + i] * bcand[i % numel_cand];
                    gmul[i] += acc;
                }
				
                break;
            }

            case FLOAT32: {
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gmul = (nnl2_float32*)grad_mult->data;
                nnl2_float32* bcand = (nnl2_float32*)data_cand->data;

                for (size_t i = 0; i < numel_mult; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_for_mult; b++) acc += gout[b * numel_mult + i] * bcand[i % numel_cand];
                    gmul[i] += acc;
                }
				
                break;
            }

            case INT32: {
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gmul = (nnl2_int32*)grad_mult->data;
                nnl2_int32* bcand = (nnl2_int32*)data_cand->data;

                for (size_t i = 0; i < numel_mult; i++) {
                    nnl2_int32 acc = 0;
                    for (size_t b = 0; b < blocks_for_mult; b++) acc += gout[b * numel_mult + i] * bcand[i % numel_cand];
                    gmul[i] += acc;
                }
				
                break;
            }

            default: NNL2_TYPE_ERROR(data_mult->dtype);
        }
    }

    if (multiplicand->requires_grad) {
        switch (data_cand->dtype) {
            case FLOAT64: {
                nnl2_float64* gout = (nnl2_float64*)grad_out->data;
                nnl2_float64* gcand = (nnl2_float64*)grad_cand->data;
                nnl2_float64* bmult = (nnl2_float64*)data_mult->data;

                for (size_t i = 0; i < numel_cand; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks_for_cand; b++) acc += gout[b * numel_cand + i] * bmult[i % numel_mult];
                    gcand[i] += acc;
                }
				
                break;
            }

            case FLOAT32: {
                nnl2_float32* gout = (nnl2_float32*)grad_out->data;
                nnl2_float32* gcand = (nnl2_float32*)grad_cand->data;
                nnl2_float32* bmult = (nnl2_float32*)data_mult->data;

                for (size_t i = 0; i < numel_cand; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_for_cand; b++) acc += gout[b * numel_cand + i] * bmult[i % numel_mult];
                    gcand[i] += acc;
                }
				
                break;
            }

            case INT32: {
                nnl2_int32* gout = (nnl2_int32*)grad_out->data;
                nnl2_int32* gcand = (nnl2_int32*)grad_cand->data;
                nnl2_int32* bmult = (nnl2_int32*)data_mult->data;

                for (size_t i = 0; i < numel_cand; i++) {
                    nnl2_int32 acc = 0;
                    for (size_t b = 0; b < blocks_for_cand; b++) acc += gout[b * numel_cand + i] * bmult[i % numel_mult];
                    gcand[i] += acc;
                }
				
                break;
            }

            default: NNL2_TYPE_ERROR(data_cand->dtype);
        }
    }
}

#endif /** NNL2_AD_MUL_BROADCASTING_BACKWARD_H **/
