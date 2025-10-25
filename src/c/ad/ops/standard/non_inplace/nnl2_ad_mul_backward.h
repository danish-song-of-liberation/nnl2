#ifndef NNL2_AD_MUL_BACKWARD_H
#define NNL2_AD_MUL_BACKWARD_H

void nnl2_ad_reverse_derivative_mul(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) {
	nnl2_tensor* grad_out_a = mul(out_tensor->grad, sumend->data);
	nnl2_tensor* grad_out_b = mul(out_tensor->grad, addend->data);
	if(addend->requires_grad) nnl2_add_inplace(addend->grad, grad_out_a);
	if(sumend->requires_grad) nnl2_add_inplace(sumend->grad, grad_out_b);
	nnl2_free_tensor(grad_out_a);
	nnl2_free_tensor(grad_out_b);
}

#endif /** NNL2_AD_MUL_BACKWARD_H **/
