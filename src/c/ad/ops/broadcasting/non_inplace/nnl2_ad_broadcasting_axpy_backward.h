#ifndef NNL2_AD_AXPY_BROADCASTING_BACKWARD_H
#define NNL2_AD_AXPY_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_axpy_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* axpyend, nnl2_ad_tensor* sumend, float multiplier) {
    nnl2_tensor* grad_out = out_tensor->grad;
    nnl2_tensor* grad_axpy = axpyend->grad;
    nnl2_tensor* grad_sum = sumend->grad;

    nnl2_tensor* data_axpy = axpyend->data;
    nnl2_tensor* data_sum = sumend->data;

    size_t numel_out = product(grad_out->shape, grad_out->rank);
    size_t numel_axpy = product(data_axpy->shape, data_axpy->rank);
    size_t numel_sum = product(data_sum->shape, data_sum->rank);

    size_t blocks_axpy = numel_out / numel_axpy;
    size_t blocks_sum = numel_out / numel_sum;

    switch (data_axpy->dtype) {
        case FLOAT64: {
            nnl2_float64* gout = (nnl2_float64*)grad_out->data;

            if (axpyend->requires_grad) {
                nnl2_float64* gax = (nnl2_float64*)grad_axpy->data;
                for (size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * (nnl2_float64)multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if (sumend->requires_grad) {
                nnl2_float64* gsum = (nnl2_float64*)grad_sum->data;
                for (size_t i = 0; i < numel_sum; i++) {
                    nnl2_float64 acc = 0.0;
                    for (size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        case FLOAT32: {
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;

            if (axpyend->requires_grad) {
                nnl2_float32* gax = (nnl2_float32*)grad_axpy->data;
                for (size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * (nnl2_float32)multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if (sumend->requires_grad) {
                nnl2_float32* gsum = (nnl2_float32*)grad_sum->data;
                for (size_t i = 0; i < numel_sum; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        case INT32: {
            nnl2_float32* gout = (nnl2_float32*)grad_out->data;

            if (axpyend->requires_grad) {
                nnl2_float32* gax = (nnl2_float32*)grad_axpy->data;
                for (size_t i = 0; i < numel_axpy; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_axpy; b++) {
                        size_t j = b * numel_axpy + i;
                        acc += gout[j] * multiplier;
                    }
					
                    gax[i] += acc;
                }
            }

            if (sumend->requires_grad) {
                nnl2_float32* gsum = (nnl2_float32*)grad_sum->data;
                for (size_t i = 0; i < numel_sum; i++) {
                    nnl2_float32 acc = 0.0f;
                    for (size_t b = 0; b < blocks_sum; b++) {
                        size_t j = b * numel_sum + i;
                        acc += gout[j];
                    }
					
                    gsum[i] += acc;
                }
            }

            break;
        }

        default: NNL2_TYPE_ERROR(data_axpy->dtype);
    }
}

#endif /** NNL2_AD_AXPY_BROADCASTING_BACKWARD_H **/
