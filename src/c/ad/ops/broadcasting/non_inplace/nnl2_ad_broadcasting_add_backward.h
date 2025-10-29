#ifndef NNL2_AD_ADD_BROADCASTING_BACKWARD_H
#define NNL2_AD_ADD_BROADCASTING_BACKWARD_H

void nnl2_ad_reverse_derivative_add_broadcasting(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* summand, nnl2_ad_tensor* sumend) {
    if(summand->requires_grad) nnl2_add_inplace(summand->grad, out_tensor->grad);

    // todo
	
	if(sumend->requires_grad) {
		nnl2_tensor* summed_grad = nnl2_sum_with_axis(summand->data, 0);
		nnl2_add_inplace(sumend->grad, summed_grad);
	}
}

#endif /** NNL2_AD_ADD_BROADCASTING_BACKWARD_H **/
