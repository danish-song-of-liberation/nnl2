#ifndef NNL2_AD_ADD_BACKWARD_DERIVATIVE_H
#define NNL2_AD_ADD_BACKWARD_DERIVATIVE_H

void nnl2_ad_reverse_derivative_axpy(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend, float multiplier) {
	if(addend->requires_grad) {
        nnl2_tensor* scaled_grad = scale(out_tensor->grad, multiplier, true);
        nnl2_add_inplace(addend->grad, scaled_grad);
        nnl2_free_tensor(scaled_grad);
    }
	
	if (sumend->requires_grad) {
        addinplace(sumend->grad, out_tensor->grad);
    }
}

#endif /** NNL2_AD_ADD_BACKWARD_DERIVATIVE_H **/
