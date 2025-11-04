#ifndef NNL2_AD_SUB_DECF_INPLACE_H
#define NNL2_AD_SUB_DECF_INPLACE_H 

void nnl2_ad_sub_decf_inplace(nnl2_ad_tensor* tensor, void* dec, bool retain_graph) {
    if(tensor->requires_grad && retain_graph) {
        NNL2_AD_INPLACE_FATAL("-= (.- in-place) (correspondence)", tensor);
    }
    
	sub_decf_inplace(tensor->data, dec);
}

#endif /** NNL2_AD_SUB_DECF_INPLACE_H **/