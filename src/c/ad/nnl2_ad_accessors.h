#ifndef NNL2_AD_ACCESSORS_H
#define NNL2_AD_ACCESSORS_H

/** @brief 
 * Retrieves the underlying data tensor from 
 * an automatic differentiation tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return nnl2_tensor*
 * Pointer to the underlying data tensor
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
nnl2_tensor* nnl2_ad_get_data(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->data;
}

/** @brief 
 * Checks if the automatic differentiation tensor is a leaf node
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return bool
 * True if the tensor is a leaf node, false otherwise
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
bool nnl2_ad_get_leaf(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->is_leaf;
}

/** @brief 
 * Checks if the automatic differentiation 
 * tensor requires gradient computation
 *
 ** @param ad_tensor 
 * Pointer to the automatic differentiation tensor
 *
 ** @return bool
 * True if the tensor requires gradient computation, false otherwise
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
bool nnl2_ad_get_requires_grad(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->requires_grad;
}

int32_t* nnl2_ad_get_shape(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->data->shape;
}

int32_t nnl2_ad_get_rank(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->data->rank;
}

size_t nnl2_ad_get_num_roots(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->num_roots;
}

nnl2_ad_tensor** nnl2_ad_get_roots(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->roots;
}

void nnl2_ad_roots_setter(nnl2_ad_tensor* ad_tensor, nnl2_ad_tensor** new_roots, size_t new_num_roots) {
	ad_tensor->roots = new_roots;
	ad_tensor->num_roots = new_num_roots;
}

nnl2_tensor_type nnl2_ad_get_dtype_as_data(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->data->dtype;
}

nnl2_tensor_type nnl2_ad_get_dtype_as_grad(nnl2_ad_tensor* ad_tensor) {
	return ad_tensor->grad->dtype;;
}

nnl2_tensor* nnl2_ad_get_grad(nnl2_ad_tensor* ad_tensor) {
	if(!ad_tensor->requires_grad) {
		if(ad_tensor->name != NULL) {
			NNL2_ERROR("For tensor %x (namely %s), an attempt was made to obtain gradients when it did not require them. Did you forget`:requires-grad t` ?", ad_tensor, ad_tensor->name);
		} else {
			NNL2_ERROR("For tensor %x, an attempt was made to obtain gradients when it did not require them. Did you forget`:requires-grad t` ?", ad_tensor);
		}
	}
	
	if(ad_tensor->grad_initialized) {
		return ad_tensor->grad;
	} else {
		NNL2_ERROR("An attempt was made to obtain an uninitialized gradient. Before getting the gradient, first perform backpropagation");
		return NULL;
	}
}

#endif /** NNL2_AD_ACCESSORS_H **/
