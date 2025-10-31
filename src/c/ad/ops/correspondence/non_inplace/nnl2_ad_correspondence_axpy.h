#ifndef NNL2_AD_AXPF_H
#define NNL2_AD_AXPF_H

void nnl2_ad_reverse_backward_axpf(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_axpf(tensor, tensor->roots[0], tensor->extra_correspondence, tensor->extra_multiplier);
}	

nnl2_ad_tensor* nnl2_ad_axpf(nnl2_ad_tensor* summand, void* sumend, float alpha, nnl2_ad_mode ad_mode) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    
    result->data = axpf(summand->data, sumend, alpha);
    result->grad = nnl2_empty(summand->data->shape, summand->data->rank, summand->data->dtype);
    result->num_roots = 1;
    result->roots = (nnl2_ad_tensor**)malloc(sizeof(nnl2_ad_tensor*));
    result->roots[0] = summand;
    result->extra_correspondence = sumend;
    result->extra_multiplier = alpha;
    result->requires_grad = summand->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_axpf; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_AXPF_H **/
