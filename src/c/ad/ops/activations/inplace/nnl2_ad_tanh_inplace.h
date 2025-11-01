#ifndef NNL2_AD_TANH_INPLACE_H
#define NNL2_AD_TANH_INPLACE_H

void nnl2_ad_inplace_tanh(nnl2_ad_tensor* ad_tensor, bool approx) {
    if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
        NNL2_AD_INPLACE_FATAL(".tanh!", ad_tensor);
    }

    tanhinplace(ad_tensor->data, approx);
}

#endif /** NNL2_AD_TANH_INPLACE_H **/
