#ifndef NNL2_AD_STEP_H
#define NNL2_AD_STEP_H

nnl2_tensor* nnl2_ad_step(nnl2_ad_tensor* ad_tensor, float learning_rate) {
	return axpy(ad_tensor->data, ad_tensor->grad, -learning_rate);
}

#endif /** NNL2_AD_STEP_H **/
