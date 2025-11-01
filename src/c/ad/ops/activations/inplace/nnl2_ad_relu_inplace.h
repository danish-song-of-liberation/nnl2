#ifndef NNL2_AD_RELU_INPLACE_H
#define NNL2_AD_RELU_INPLACE_H

void nnl2_ad_inplace_relu(nnl2_ad_tensor* ad_tensor) {
    if(ad_tensor->requires_grad) {
        NNL2_AD_INPLACE_FATAL(".relu!", ad_tensor);
    }

    reluinplace(ad_tensor->data);
}

#endif /** NNL2_AD_RELU_INPLACE_H **/
