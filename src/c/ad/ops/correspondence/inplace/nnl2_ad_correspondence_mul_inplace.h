#ifndef NNL2_AD_MUL_MULF_INPLACE_H
#define NNL2_AD_MUL_MULF_INPLACE_H

void nnl2_ad_mul_mulf_inplace(nnl2_ad_tensor* tensor, void* multiplier) {
    if(tensor->requires_grad) {
        NNL2_AD_INPLACE_FATAL("*= (.* in-place) (correspondence)", tensor);
    }
     
	mul_mulf_inplace(tensor->data, multiplier);
}

#endif /** NNL2_AD_MUL_MULF_INPLACE_H **/
