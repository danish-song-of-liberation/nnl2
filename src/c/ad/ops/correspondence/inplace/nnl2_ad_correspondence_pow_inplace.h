#ifndef NNL2_AD_POW_POWF_INPLACE_H
#define NNL2_AD_POW_POWF_INPLACE_H

void nnl2_ad_pow_powf_inplace(nnl2_ad_tensor* tensor, void* exponent, bool retain_graph) {
    if(tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".max! (.max in-place) (correspondence)", tensor);
    }
    
    pow_powf_inplace(tensor->data, exponent);
}

#endif /** NNL2_AD_POW_POWF_INPLACE_H **/
