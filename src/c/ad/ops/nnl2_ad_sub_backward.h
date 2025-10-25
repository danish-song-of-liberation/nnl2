#ifndef NNL2_AD_SUB_BACKWARD_H
#define NNL2_AD_SUB_BACKWARD_H

void nnl2_ad_reverse_derivative_sub(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend) {
	nnl2_add_inplace(minuend->grad, out_tensor->grad);
	subinplace(subtrahend->grad, out_tensor->grad);
}

#endif /** NNL2_AD_SUB_BACKWARD_H **/