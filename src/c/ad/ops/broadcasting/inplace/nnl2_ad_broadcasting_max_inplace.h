#ifndef NNL2_AD_MAX_BROADCASTING_INPLACE_H
#define NNL2_AD_MAX_BROADCASTING_INPLACE_H

void nnl2_ad_max_broadcasting_inplace(nnl2_ad_tensor* tensor_a, nnl2_ad_tensor* tensor_b) {
    if(tensor_a->requires_grad) {
        NNL2_AD_INPLACE_FATAL(".max! (.max in-place) (broadcasting)", tensor_a);
    }
    
    max_broadcasting_inplace(tensor_a->data, tensor_b->data);
}

#endif /** NNL2_AD_MAX_BROADCASTING_INPLACE_H **/
