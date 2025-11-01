#ifndef NNL2_AD_MUL_INPLACE_H
#define NNL2_AD_MUL_INPLACE_H

void nnl2_ad_mul_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand) {
    if(result->requires_grad) {
        NNL2_AD_INPLACE_FATAL("*= (.* in-place)", result);
    }
    
    mulinplace(result->data, operand->data);
}

#endif /** NNL2_AD_MUL_INPLACE_H **/
