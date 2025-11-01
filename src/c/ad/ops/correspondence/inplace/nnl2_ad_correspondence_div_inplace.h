#ifndef NNL2_AD_DIV_DIVF_INPLACE_H
#define NNL2_AD_DIV_DIVF_INPLACE_H

void nnl2_ad_div_divf_inplace(nnl2_ad_tensor* tensor, void* divisor) {
    if(tensor->requires_grad) {
        NNL2_AD_INPLACE_FATAL("/! (./ in-place) (correspondence)", tensor);
    }

	div_divf_inplace(tensor->data, divisor);
}

#endif /** NNL2_AD_DIV_DIVF_INPLACE_H **/
