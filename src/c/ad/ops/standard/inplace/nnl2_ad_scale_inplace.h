#ifndef NNL2_AD_SCALE_INPLACE_H
#define NNL2_AD_SCALE_INPLACE_H

void nnl2_ad_inplace_scale(nnl2_ad_tensor* ad_tensor, nnl2_float32 multiplier, bool retain_graph) {
    if(ad_tensor->requires_grad && retain_graph) {	
        NNL2_AD_INPLACE_FATAL("scale!", ad_tensor);
    }
    
    scaleinplace(ad_tensor->data, multiplier);
}

#endif /** NNL2_AD_SCALE_INPLACE_H **/
