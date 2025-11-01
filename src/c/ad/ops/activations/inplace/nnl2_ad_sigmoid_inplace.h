#ifndef NNL2_AD_SIGMOID_INPLACE_H
#define NNL2_AD_SIGMOID_INPLACE_H

void nnl2_ad_inplace_sigmoid(nnl2_ad_tensor* ad_tensor, bool approx) {
    if(ad_tensor->requires_grad && !ad_tensor->is_leaf) {
        NNL2_AD_INPLACE_FATAL(".sigmoid!", ad_tensor);
    }
    
    sigmoidinplace(ad_tensor->data, approx);
}

#endif /** NNL2_AD_SIGMOID_INPLACE_H **/
