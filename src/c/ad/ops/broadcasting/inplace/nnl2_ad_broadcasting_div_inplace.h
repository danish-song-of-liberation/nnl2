#ifndef NNL2_AD_DIV_BROADCASTING_INPLACE_H
#define NNL2_AD_DIV_BROADCASTING_INPLACE_H  

void nnl2_ad_div_broadcasting_inplace(nnl2_ad_tensor* dividend, nnl2_ad_tensor* divisor) {
    if(dividend->requires_grad) {
		NNL2_AD_INPLACE_FATAL("/! (./ in-place) (broadcasting)", dividend);
    }
    
    div_broadcasting_inplace(dividend->data, divisor->data);
}

#endif /** NNL2_AD_DIV_BROADCASTING_INPLACE_H **/
