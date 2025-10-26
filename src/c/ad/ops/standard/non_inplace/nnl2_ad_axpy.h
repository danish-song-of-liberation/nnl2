#ifndef NNL2_AD_AXPY_H
#define NNL2_AD_AXPY_H

void nnl2_ad_reverse_backward_axpy(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_axpy(tensor, tensor->roots[0], tensor->roots[1], tensor->extra_multiplier);
}	

nnl2_ad_tensor* nnl2_ad_axpy(nnl2_ad_tensor* axpyend, nnl2_ad_tensor* sumend, float multiplier, nnl2_ad_mode ad_mode) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    
    result->data = axpy(axpyend->data, sumend->data, multiplier);
    result->grad = nnl2_empty(axpyend->data->shape, axpyend->data->rank, axpyend->data->dtype);
    result->num_roots = 2;
    result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
    result->roots[0] = axpyend;
    result->roots[1] = sumend;
    result->requires_grad = axpyend->requires_grad || sumend->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    result->extra_multiplier = multiplier; 
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_axpy; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_AXPY_H **/
