#ifndef NNL2_AD_MUL_H
#define NNL2_AD_MUL_H

void nnl2_ad_reverse_backward_mul(nnl2_ad_tensor* tensor) {
    nnl2_ad_reverse_derivative_mul(tensor, tensor->roots[0], tensor->roots[1]);
}   

nnl2_ad_tensor* nnl2_ad_mul(nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier, nnl2_ad_mode ad_mode) {
    nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
    
    result->data = mul(multiplicand->data, multiplier->data);
    result->grad = nnl2_empty(multiplicand->data->shape, multiplicand->data->rank, multiplicand->data->dtype);
    result->num_roots = 2;
    result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
    result->roots[0] = multiplicand;
    result->roots[1] = multiplier;
    result->requires_grad = multiplicand->requires_grad || multiplier->requires_grad;
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->grad_initialized = false;
    result->is_leaf = false; 
    result->name = NULL;
	result->ts_type = nnl2_type_ad;
	
    switch(ad_mode) {
        case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_mul; break;
        
        default: {
            NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
            nnl2_free_ad_tensor(result);
            return NULL;
        }
    }
    
    return result;
}

#endif /** NNL2_AD_MUL_H **/
