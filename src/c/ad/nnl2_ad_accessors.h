#ifndef NNL2_AD_ACCESSORS_H
#define NNL2_AD_ACCESSORS_H

nnl2_tensor* nnl2_ad_get_data(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->data;
}

bool nnl2_ad_get_leaf(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->is_leaf;
}

bool nnl2_ad_get_requires_grad(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->requires_grad;
}

#endif /** NNL2_AD_ACCESSORS_H **/
