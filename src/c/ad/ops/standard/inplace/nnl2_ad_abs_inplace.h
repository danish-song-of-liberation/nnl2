#ifndef NNL2_AD_ABS_INPLACE_H
#define NNL2_AD_ABS_INPLACE_H

void nnl2_ad_inplace_abs(nnl2_ad_tensor* ad_tensor, bool retain_graph) {
	if(ad_tensor->requires_grad && retain_graph) {
		NNL2_AD_INPLACE_FATAL(".abs! (.abs in-place)", ad_tensor);
	}
	
    absinplace(ad_tensor->data);
}

#endif /** NNL2_AD_ABS_INPLACE_H **/
