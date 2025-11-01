#ifndef NNL2_AD_LEAKYRELU_INPLACE_H
#define NNL2_AD_LEAKYRELU_INPLACE_H

void nnl2_ad_inplace_leakyrelu(nnl2_ad_tensor* ad_tensor, nnl2_float32 alpha) {
    if(ad_tensor->requires_grad) {
        NNL2_AD_INPLACE_FATAL(".leaky-relu!", ad_tensor);
    }

    leakyreluinplace(ad_tensor->data, alpha);
}

#endif /** NNL2_AD_LEAKYRELU_INPLACE_H **/
