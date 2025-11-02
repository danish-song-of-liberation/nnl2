#ifndef NNL2_AD_NEG_BACKWARD_H
#define NNL2_AD_NEG_BACKWARD_H

void nnl2_ad_reverse_derivative_neg(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor) {
	nnl2_tensor* neg_grad = nnl2_neg(out_tensor->grad);
    nnl2_add_inplace(ad_tensor->grad, neg_grad);
    nnl2_free_tensor(neg_grad);
}

#endif /** NNL2_AD_NEG_BACKWARD_H **/
