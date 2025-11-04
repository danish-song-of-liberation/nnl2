#ifndef NNL2_AD_AXPY_BROADCASTING_INPLACE_H
#define NNL2_AD_AXPY_BROADCASTING_INPLACE_H

void nnl2_ad_axpy_broadcasting_inplace(nnl2_ad_tensor* sumend, nnl2_ad_tensor* axpyend, float multiplier, bool retain_graph) {
    if(sumend->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("axpy! (broadcasting)", sumend);
    }
    
    axpy_broadcasting_inplace(sumend->data, axpyend->data, multiplier);
}

#endif /** NNL2_AD_AXPY_BROADCASTING_INPLACE_H **/
