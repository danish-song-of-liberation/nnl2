#ifndef NNL2_AD_ADD_CORRESPONDENCE_H
#define NNL2_AD_ADD_CORRESPONDENCE_H

void nnl2_ad_reverse_backward_add_correspondence(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_add_correspondence(tensor, tensor->roots[0], tensor->extra_correspondence);
}	

nnl2_ad_tensor* nnl2_ad_add_correspondence(nnl2_ad_tensor* tensor, void* inc, nnl2_ad_mode ad_mode, bool track_graph) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));

    result->data = add_incf(tensor->data, inc);
    result->grad = nnl2_empty(tensor->data->shape, tensor->data->rank, tensor->data->dtype);

    if (track_graph) {
        result->num_roots = 1;
        result->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
        result->roots[0] = tensor;
    } else {
        result->num_roots = 0;
        result->roots = NULL;
    }

    result->extra_correspondence = inc;
    result->requires_grad = tensor->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;

    switch (ad_mode) {
        case nnl2_ad_reverse_mode:
            result->backward_fn = nnl2_ad_reverse_backward_add_correspondence;
            break;

        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }

    return result;
}

#endif /** NNL2_AD_ADD_CORRESPONDENCE_H **/
