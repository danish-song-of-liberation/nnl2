#ifndef NNL2_AD_AXPY_INPLACE_H
#define NNL2_AD_AXPY_INPLACE_H

void nnl2_ad_inplace_axpy(nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend, float multiplier) {
    if(summand->requires_grad) {
		NNL2_AD_INPLACE_FATAL("axpy", summand);
    }

    axpy_inplace(summand->data, sumend->data, multiplier);
}

#endif /** NNL2_AD_AXPY_INPLACE_H **/
