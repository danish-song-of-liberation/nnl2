#ifndef NNL2_AD_GENS_H
#define NNL2_AD_GENS_H

/** @file nnl2_ad_gens.h
 ** @copyright MIT
 ** @date 2025
 *
 * The file contains functions for creating 
 * AD-tensors, including zeros, ones, full, etc.
 *
 ** Filepath: nnl2/src/c/ad/nnl2_adgens.h
 **/
 
nnl2_ad_tensor* nnl2_ad_full(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, void* fill_with) {
	nnl2_ad_tensor* ad_tensor = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!ad_tensor) {
			NNL2_MALLOC_ERROR();
			return NULL; 
		}
	#endif
	
	ad_tensor->data = nnl2_full(shape, rank, dtype, &fill_with);
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!ad_tensor->data) {
			NNL2_TENSOR_ERROR("custom");
			free(ad_tensor);
			return NULL; 
		}
	#endif
	
	if(requires_grad) {
		ad_tensor->grad = nnl2_empty(shape, rank, dtype);
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!ad_tensor->grad) {
				NNL2_TENSOR_ERROR("non-initialized memory");
				nnl2_free_tensor(ad_tensor->data);
				free(ad_tensor);
				return NULL;
			}
		#endif
	} else {
		ad_tensor->grad = NULL;
	}
	
	ad_tensor->requires_grad = requires_grad;
	ad_tensor->backward_fn = NULL;
	ad_tensor->roots = NULL;
    ad_tensor->num_roots = 0;
    ad_tensor->grad_computed = NULL;
	ad_tensor->is_leaf = true;
	
	if(name) {
		ad_tensor->name = malloc(strlen(name) + 1);
		
		if (ad_tensor->name) {
			strcpy(ad_tensor->name, name);
		} else {
			NNL2_WARN("Failed to allocate memory for the AD tensor name (the name will be replaced with NULL)");
			ad_tensor->name = NULL;
		}
	} else {
		ad_tensor->name = NULL;
	}
		
	return ad_tensor;
}	

nnl2_ad_tensor* nnl2_ad_zeros(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
	nnl2_float32 zero = 0.0f;
	return nnl2_ad_full(shape, rank, dtype, requires_grad, name, &zero);
}

nnl2_ad_tensor* nnl2_ad_ones(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
	nnl2_float32 one = 1.0f;
	return nnl2_ad_full(shape, rank, dtype, requires_grad, name, &one);
}
 
#endif /** NNL2_AD_GENS_H **/
