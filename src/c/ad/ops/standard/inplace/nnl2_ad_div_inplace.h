#ifndef NNL2_AD_DIV_INPLACE_H
#define NNL2_AD_DIV_INPLACE_H

void nnl2_ad_div_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand) {
    if(result->requires_grad) {
        NNL2_AD_INPLACE_FATAL("/! (division in-place)", result);
    }
    
    divinplace(result->data, operand->data);
}

#endif /** NNL2_AD_DIV_INPLACE_H **/
