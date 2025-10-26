#ifndef NNL2_AD_MAX_H
#define NNL2_AD_MAX_H

void nnl2_ad_reverse_backward_max(nnl2_ad_tensor* tensor) {
	nnl2_ad_reverse_derivative_max(tensor, tensor->roots[0], tensor->roots[1]);
}	

nnl2_ad_tensor* nnl2_ad_max(nnl2_ad_tensor* a, nnl2_ad_tensor* b, nnl2_ad_mode ad_mode) {
	nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
	
	result->data = nnl2_max(a->data, b->data);
	result->grad = nnl2_empty(a->data->shape, a->data->rank, a->data->dtype);
	result->num_roots = 2;
	result->roots = (nnl2_ad_tensor**)malloc(2 * sizeof(nnl2_ad_tensor*));
	result->roots[0] = a;
	result->roots[1] = b;
	result->requires_grad = a->requires_grad || b->requires_grad;
	result->magic_number = TENSOR_MAGIC_ALIVE;
	result->grad_initialized = false;
	result->is_leaf = false; 
    result->name = NULL;
	result->ts_type = nnl2_type_ad;
	
	switch(ad_mode) {
		case nnl2_ad_reverse_mode: result->backward_fn = nnl2_ad_reverse_backward_max; break;
		
		default: {
			NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
			nnl2_free_ad_tensor(result);
			return NULL;
		}
	}
	
	return result;
}

#endif /** NNL2_AD_MAX_H **/
