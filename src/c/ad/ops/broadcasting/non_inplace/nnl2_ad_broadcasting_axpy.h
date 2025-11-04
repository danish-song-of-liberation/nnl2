#ifndef NNL2_AD_AXPY_BROADCASTING_H
#define NNL2_AD_AXPY_BROADCASTING_H

void nnl2_ad_reverse_backward_axpy_broadcasting(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_axpy_broadcasting(tensor, tensor->roots[0], tensor->roots[1], tensor->extra_multiplier);
}	

nnl2_ad_tensor* nnl2_ad_axpy_broadcasting(nnl2_ad_tensor* axpyend, nnl2_ad_tensor* sumend, float multiplier, nnl2_ad_mode ad_mode, bool track_graph) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));

    result->data = axpy_broadcasting(axpyend->data, sumend->data, multiplier);
    result->grad = nnl2_empty(axpyend->data->shape, axpyend->data->rank, axpyend->data->dtype);
    
    if (track_graph) {
        result->num_roots = 2;
        result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
        result->roots[0] = axpyend;
        result->roots[1] = sumend;
    } else {
        result->num_roots = 0;
        result->roots = NULL;
    }

    result->extra_multiplier = multiplier;
    result->requires_grad = axpyend->requires_grad || sumend->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;

    switch (ad_mode) {
        case nnl2_ad_reverse_mode:
            result->backward_fn = nnl2_ad_reverse_backward_axpy_broadcasting;
            break;

        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }

    return result;
}

#endif /** NNL2_AD_AXPY_BROADCASTING_H **/
