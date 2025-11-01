#ifndef NNL2_AD_LOG_BACKWARD_H
#define NNL2_AD_LOG_BACKWARD_H

void nnl2_ad_reverse_derivative_scale(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* ad_tensor, float multiplier) {
	// todo remake
	nnl2_tensor* scaled_grad = scale(out_tensor->grad, multiplier, true);
    nnl2_add_inplace(ad_tensor->grad, scaled_grad);
    nnl2_free_tensor(scaled_grad);
}

#endif /** NNL2_AD_LOG_BACKWARD_H **/
