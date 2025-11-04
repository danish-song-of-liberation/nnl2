#ifndef NNL2_AD_MAX_MAXF_INPLACE_H
#define NNL2_AD_MAX_MAXF_INPLACE_H

void nnl2_ad_max_maxf_inplace(nnl2_ad_tensor* tensor, void* threshold, bool retain_graph) {
    if(tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".max! (.max in-place) (correspondence)", tensor);
    }
 
    max_maxf_inplace(tensor->data, threshold);
}

#endif /** NNL2_AD_MAX_MAXF_INPLACE_H **/
