#ifndef NNL2_AD_BACKPROPAGATION_H
#define NNL2_AD_BACKPROPAGATION_H

#define nnl2_ad_bp(tensor) nnl2_ad_backpropagation(tensor) // Reduction

void nnl2_ad_backpropagation(nnl2_ad_tensor* tensor) {
	int topo_size;
    nnl2_ad_tensor** topo = nnl2_ad_build_topo(tensor, &topo_size);
	
	float grad_init = 1.0f;
	
	inplace_fill(tensor->grad, &grad_init, tensor->data->dtype);
	// bool success = inplace_fill(tensor->grad, &grad_init, tensor->data->dtype);
	
	for (int i = topo_size - 1; i >= 0; i--) {
		if (topo[i]->requires_grad) {
            topo[i]->backward_fn(topo[i]);
        }
	}
	
	NNL2_WARN("Do not add tenosr freeing in nnl2_ad_backpropagation in the future");
}

#endif /** NNL2_AD_BACKPROPAGATION_H **/	
