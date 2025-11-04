#ifndef NNL2_AD_LOG_INPLACE_H
#define NNL2_AD_LOG_INPLACE_H

void nnl2_ad_inplace_log(nnl2_ad_tensor* ad_tensor, bool retain_graph) {
    if(ad_tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL(".log! (.log in-place)", ad_tensor);
    }
    
    loginplace(ad_tensor->data);
}

#endif /** NNL2_AD_LOG_INPLACE_H **/
