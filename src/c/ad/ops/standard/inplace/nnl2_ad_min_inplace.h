#ifndef NNL2_AD_MIN_INPLACE_H
#define NNL2_AD_MIN_INPLACE_H

void nnl2_ad_min_inplace(nnl2_ad_tensor* result, nnl2_ad_tensor* operand, bool retain_graph) {
    if(result->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".min! (.min in-place)", result);
    }
    
    mininplace(result->data, operand->data);
}

#endif /** NNL2_AD_MIN_INPLACE_H **/
