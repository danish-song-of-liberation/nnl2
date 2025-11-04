#ifndef NNL2_AD_MIN_MINF_INPLACE_H
#define NNL2_AD_MIN_MINF_INPLACE_H

void nnl2_ad_min_minf_inplace(nnl2_ad_tensor* tensor, void* threshold, bool retain_graph) {
    if(tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".min! (.min in-place) (correspondence)", tensor);
    }
  
    min_minf_inplace(tensor->data, threshold);
}

#endif /** NNL2_AD_MIN_MINF_INPLACE_H **/
