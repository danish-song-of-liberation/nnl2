#ifndef NNL2_AD_MAX_INPLACE_H
#define NNL2_AD_MAX_INPLACE_H

void nnl2_ad_max_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand) {
    if(result->requires_grad) {
        NNL2_AD_INPLACE_FATAL(".max! (.max in-place)", result);
    }
    
    maxinplace(result->data, operand->data);
}

#endif /** NNL2_AD_MAX_INPLACE_H **/
