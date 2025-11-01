#ifndef NNL2_AD_ADD_INPLACE_H
#define NNL2_AD_ADD_INPLACE_H

void nnl2_ad_add_inplace(nnl2_ad_tensor* summand, nnl2_ad_tensor* addend) {
	if(summand->requires_grad) {
        NNL2_AD_INPLACE_FATAL("+= (Addition in-place)", summand);
    }
	
	addinplace(summand->data, addend->data);
}

#endif /** NNL2_AD_ADD_INPLACE_H **/
