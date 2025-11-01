#ifndef NNL2_AD_SUB_BROADCASTING_INPLACE_H
#define NNL2_AD_SUB_BROADCASTING_INPLACE_H

void nnl2_ad_sub_broadcasting_inplace(nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend) {
    if(minuend->requires_grad) {
        NNL2_AD_INPLACE_FATAL("-= (.- in-place) (broadcasting)", minuend);
    }
    
    sub_broadcasting_inplace(minuend->data, subtrahend->data);
}

#endif /** NNL2_AD_SUB_BROADCASTING_INPLACE_H **/
