#ifndef NNL2_AD_MIN_CORRESPONDENCE_H
#define NNL2_AD_MIN_CORRESPONDENCE_H

void nnl2_ad_reverse_backward_min_correspondence(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_min_correspondence(tensor, tensor->roots[0], tensor->extra_correspondence);
}   

nnl2_ad_tensor* nnl2_ad_min_correspondence(nnl2_ad_tensor* tensor, void* threshold, nnl2_ad_mode ad_mode, bool track_graph) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    
    result->data = min_minf(tensor->data, threshold);
    result->grad = nnl2_empty(tensor->data->shape, tensor->data->rank, tensor->data->dtype);

    if (track_graph) {
        result->num_roots = 1;
        result->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
        result->roots[0] = tensor;
    } else {
        result->num_roots = 0;
        result->roots = NULL;
    }

    result->extra_correspondence = threshold;
    result->requires_grad = tensor->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    
    switch (ad_mode) {
        case nnl2_ad_reverse_mode:
            result->backward_fn = nnl2_ad_reverse_backward_min_correspondence;
            break;

        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_MIN_CORRESPONDENCE_H **/
