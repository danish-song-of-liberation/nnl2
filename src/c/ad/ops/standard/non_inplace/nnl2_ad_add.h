#ifndef NNL2_AD_ADD_H
#define NNL2_AD_ADD_H

void nnl2_ad_reverse_backward_add(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_add(tensor, tensor->roots[0], tensor->roots[1]);
}

nnl2_ad_tensor* nnl2_ad_add(nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend, nnl2_ad_mode ad_mode, bool track_graph) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));

    result->data = nnl2_add(addend->data, sumend->data);
    result->grad = nnl2_empty(addend->data->shape, addend->data->rank, addend->data->dtype);

    if (track_graph) {
        result->num_roots = 2;
        result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
        result->roots[0] = addend;
        result->roots[1] = sumend;
    } else {
        result->num_roots = 0;
        result->roots = NULL;
    }

    result->requires_grad = addend->requires_grad || sumend->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false;
    result->name = NULL;
    result->ts_type = nnl2_type_ad;

    switch (ad_mode) {
        case nnl2_ad_reverse_mode:
            result->backward_fn = nnl2_ad_reverse_backward_add;
            break;

        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }

    return result;
}

#endif /** NNL2_AD_ADD_H **/
