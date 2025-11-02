#ifndef NNL2_AD_AXPF_BACKWARD_H
#define NNL2_AD_AXPF_BACKWARD_H

void nnl2_ad_reverse_derivative_axpf(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* summand_tensor, void* sumend, float alpha) {
	(void)sumend;
	(void)alpha;
	
    addinplace(summand_tensor->grad, out_tensor->grad);
}

#endif /** NNL2_AD_AXPF_BACKWARD_H **/
