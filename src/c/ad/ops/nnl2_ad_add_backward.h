#ifndef NNL2_AD_ADD_BACKWARD_H
#define NNL2_AD_ADD_BACKWARD_H

void nnl2_ad_reverse_derivative_add(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) {
	nnl2_add_inplace(addend->grad, out_tensor->grad);
	nnl2_add_inplace(sumend->grad, out_tensor->grad);
}

#endif /** NNL2_AD_ADD_BACKWARD_H **/
