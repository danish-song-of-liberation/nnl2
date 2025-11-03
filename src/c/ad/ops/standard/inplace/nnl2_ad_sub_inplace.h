#ifndef NNL2_AD_SUB_INPLACE_H
#define NNL2_AD_SUB_INPLACE_H

void nnl2_ad_sub_inplace(nnl2_ad_tensor* minuend, nnl2_ad_tensor* subtrahend, bool track_graph) {
    if(minuend->requires_grad && track_graph) {
        NNL2_AD_INPLACE_FATAL("-= (.- in-place)", minuend);
    }
    
    subinplace(minuend->data, subtrahend->data);
}

#endif /** NNL2_AD_SUB_INPLACE_H **/