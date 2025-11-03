#ifndef NNL2_AD_BACKPROPAGATION_H
#define NNL2_AD_BACKPROPAGATION_H

void nnl2_ad_backpropagation(nnl2_ad_tensor* tensor, bool retain_graph) {
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
	
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward_fn) {
            topo[i]->backward_fn(topo[i]); 			
        }
    }
	
	if(!retain_graph) nnl2_ad_clear_graph(topo, topo_size);
    
    free(topo);
}

#endif /** NNL2_AD_BACKPROPAGATION_H **/	
