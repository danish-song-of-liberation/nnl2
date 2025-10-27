#ifndef NNL2_AD_COPY_H
#define NNL2_AD_COPY_H

nnl2_ad_tensor* nnl2_ad_copy(nnl2_ad_tensor* ad_tensor, nnl2_tensor_type dtype) {
	nnl2_ad_tensor* tensor_copy = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
	tensor_copy->data = nnl2_copy(ad_tensor->data, dtype);
	tensor_copy->grad_initialized = ad_tensor->grad_initialized;
	
	if(ad_tensor->grad_initialized) {
		tensor_copy->grad = nnl2_copy(ad_tensor->grad, dtype);
		tensor_copy->grad_initialized = true;
	} else {
		tensor_copy->grad_initialized = false;
	}

	tensor_copy->ts_type = nnl2_type_ad;
	tensor_copy->is_leaf = ad_tensor->is_leaf;
	tensor_copy->magic_number = TENSOR_MAGIC_ALIVE;
	
	if(ad_tensor->num_roots > 0) {
		tensor_copy->num_roots = ad_tensor->num_roots;
		tensor_copy->roots = (nnl2_ad_tensor**)malloc(tensor_copy->num_roots * sizeof(nnl2_ad_tensor*));
		for(size_t i = 0; i < tensor_copy->num_roots; i++) {
			tensor_copy->roots[i] = ad_tensor->roots[i];
		}
	} else {
		tensor_copy->num_roots = 0;
		tensor_copy->roots = NULL;
	}
	
	if(ad_tensor->name) {
		tensor_copy->name = malloc(strlen(ad_tensor->name) + 1);
		strcpy(tensor_copy->name, ad_tensor->name);
	} else {
		tensor_copy->name = NULL;
	}
	
	return tensor_copy;
}

#endif
