#ifndef NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H
#define NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H

#define nnl2_ad_bptt(tensor) nnl2_ad_backpropagation_through_time(tensor)

void nnl2_ad_backpropagation_through_time(nnl2_ad_tensor* tensor) {
    nnl2_ad_tensor* leaf_tensor = nnl2_ad_find_leaf(tensor);
    
    int topo_size;
    nnl2_ad_tensor** topo = nnl2_ad_build_topo(tensor, &topo_size);

    for (int i = 0; i < topo_size; i++) {
        if (topo[i]->requires_grad && !topo[i]->grad_initialized) {
            void* zero_value = nnl2_get_zero_value(topo[i]->grad->dtype);
            if (zero_value) {
                inplace_fill(topo[i]->grad, zero_value, topo[i]->grad->dtype);
                topo[i]->grad_initialized = true;
            }
        }
    }

    void* grad_init_value = nnl2_get_one_value(tensor->grad->dtype);
    if (!grad_init_value) {
        free(topo);
        return;
    }
    
    bool success = inplace_fill(tensor->grad, grad_init_value, tensor->grad->dtype);
    if (!success) {
        free(topo);
        return;
    }
    
    tensor->grad_initialized = true;

    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward_fn) {
            topo[i]->backward_fn(topo[i]);  
        }
    }
    
    tensor->grad = leaf_tensor->grad;
    
    free(topo);
}

#endif /** NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H **/
