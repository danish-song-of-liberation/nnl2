#ifndef NNL2_AD_MIN_BROADCASTING_INPLACE_H
#define NNL2_AD_MIN_BROADCASTING_INPLACE_H

void nnl2_ad_min_broadcasting_inplace(nnl2_ad_tensor* tensor_a, nnl2_ad_tensor* tensor_b, bool retain_graph) {
    if(tensor_a->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".min! (.min in-place) (broadcasting)", tensor_a);
    }
    
    min_broadcasting_inplace(tensor_a->data, tensor_b->data);
}

#endif /** NNL2_AD_MIN_BROADCASTING_INPLACE_H **/
