#ifndef NNL2_AD_STEP_INPLACE_H
#define NNL2_AD_STEP_INPLACE_H

void nnl2_ad_step_inplace(nnl2_ad_tensor* ad_tensor, float learning_rate) {
	axpy_inplace(ad_tensor->data, ad_tensor->grad, -learning_rate);
}

#endif /** NNL2_AD_STEP_INPLACE_H **/
