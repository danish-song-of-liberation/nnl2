#ifndef NNL2_AD_AXPF_INPLACE_H
#define NNL2_AD_AXPF_INPLACE_H

void nnl2_ad_axpf_inplace(nnl2_ad_tensor* summand, void* sumend, float alpha, bool retain_graph) {
    if(summand->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("axpy! (axpy in-place) (correspondence)", summand);
    }

	axpf_inplace(summand->data, sumend, alpha);
}

#endif /** NNL2_AD_AXPF_INPLACE_H **/
