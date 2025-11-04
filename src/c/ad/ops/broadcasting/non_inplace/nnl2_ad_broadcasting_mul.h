#ifndef NNL2_AD_MUL_BROADCASTING_H
#define NNL2_AD_MUL_BROADCASTING_H

void nnl2_ad_reverse_backward_mul_broadcasting(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_mul_broadcasting(tensor, tensor->roots[0], tensor->roots[1]);
}   

nnl2_ad_tensor* nnl2_ad_mul_broadcasting(
    nnl2_ad_tensor* multiplier,
    nnl2_ad_tensor* multiplicand,
    nnl2_ad_mode ad_mode,
    bool track_graph
) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));

    result->data = mul_broadcasting(multiplier->data, multiplicand->data);
    result->grad = nnl2_empty(multiplier->data->shape, multiplier->data->rank, multiplier->data->dtype);
    result->requires_grad = multiplier->requires_grad || multiplicand->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->ts_type = nnl2_type_ad;

    if (track_graph && result->requires_grad) {
        result->num_roots = 2;
        result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
        result->roots[0] = multiplier;
        result->roots[1] = multiplicand;

        switch (ad_mode) {
            case nnl2_ad_reverse_mode:
                result->backward_fn = nnl2_ad_reverse_backward_mul_broadcasting;
                break;
            default:
                NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
                nnl2_free_ad_tensor(result);
                return NULL;
        }
    } else {
        result->num_roots = 0;
        result->roots = NULL;
        result->backward_fn = NULL;
    }

    return result;
}

#endif /** NNL2_AD_MUL_BROADCASTING_H **/
