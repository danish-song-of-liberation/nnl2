#ifndef NNL2_AD_NEG_INPLACE_H
#define NNL2_AD_NEG_INPLACE_H

void nnl2_ad_neg_inplace(nnl2_ad_tensor* ad_tensor, bool retain_graph) {
	if(ad_tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".neg! (.neg in-place)", ad_tensor);
    }
	
	nnl2_neginplace(ad_tensor->data);
}

#endif /** NNL2_AD_NEG_INPLACE_H **/
