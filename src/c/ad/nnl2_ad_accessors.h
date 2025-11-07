#ifndef NNL2_AD_ACCESSORS_H
#define NNL2_AD_ACCESSORS_H

// NNL2

/** @file nnl2_ad_accessors.h
 ** @brief Automatic differentiation tensor accessor functions (for lisp)
 ** @date 2025  
 ** @copyright MIT
 **/

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
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_data, ad_tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_get_data, ad_tensor data is NULL", NULL);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
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
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_leaf, ad_tensor is NULL", false);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
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
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_requires_grad, ad_tensor is NULL", false);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->requires_grad;
}

/** @brief 
 * Retrieves the shape array of the automatic differentiation tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return int32_t*
 * Pointer to the shape array of the tensor
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
int32_t* nnl2_ad_get_shape(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_shape, ad_tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_get_shape, ad_tensor shape is NULL", NULL);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->data->shape;
}

/** @brief 
 * Retrieves the rank of the automatic differentiation tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return int32_t
 * Rank (number of dimensions) of the tensor
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
int32_t nnl2_ad_get_rank(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_rank, ad_tensor is NULL", -1);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->data->rank;
}

/** @brief 
 * Retrieves the number of root tensors in the computational graph
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return size_t
 * Number of root tensors
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
size_t nnl2_ad_get_num_roots(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_num_roots, ad_tensor is NULL", 0);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->num_roots;
}

/** @brief 
 * Retrieves the array of root tensors in the computational graph
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return nnl2_ad_tensor**
 * Pointer to the array of root tensors
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
nnl2_ad_tensor** nnl2_ad_get_roots(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_roots, ad_tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->roots, "In function nnl2_ad_get_roots, ad_tensor roots is NULL", NULL);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->roots;
}

/** @brief 
 * Sets the roots array and root count for an automatic differentiation tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor to modify
 *
 ** @param new_roots
 * Pointer to the new array of root tensors
 *
 ** @param new_num_roots
 * Number of roots in the new array
 *
 ** @return void
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
void nnl2_ad_roots_setter(nnl2_ad_tensor* ad_tensor, nnl2_ad_tensor** new_roots, size_t new_num_roots) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(ad_tensor, "In function nnl2_ad_roots_setter, ad_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(new_roots, "In function nnl2_ad_roots_setter, new_roots is NULL");
	#endif
	
	ad_tensor->roots = new_roots;
	ad_tensor->num_roots = new_num_roots;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
}

/** @brief 
 * Retrieves the data type of the underlying data tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return nnl2_tensor_type
 * Data type of the tensor's data
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
nnl2_tensor_type nnl2_ad_get_dtype_as_data(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_dtype_as_data, ad_tensor is NULL", -1);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_get_dtype_as_data, ad_tensor data is NULL", -1);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->data->dtype;
}

/** @brief 
 * Retrieves the data type of the gradient tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return nnl2_tensor_type
 * Data type of the tensor's gradient
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
nnl2_tensor_type nnl2_ad_get_dtype_as_grad(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_dtype_as_grad, ad_tensor is NULL", -1);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->grad, "In function nnl2_ad_get_dtype_as_grad, ad_tensor grad is NULL", -1);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif 
	
	return ad_tensor->grad->dtype;
}

/** @brief 
 * Retrieves the gradient tensor from an automatic differentiation tensor
 *
 ** @param ad_tensor
 * Pointer to the automatic differentiation tensor
 *
 ** @return nnl2_tensor*
 * Pointer to the gradient tensor, or NULL if gradients are not available
 *
 ** @exception NNL2Error
 * If tensor does not require gradients
 *
 ** @exception NNL2Error
 * If gradients are not initialized (backward pass not performed)
 *
 ** @note 
 * This is a CFFI wrapper function for Lisp FFI integration
 */
nnl2_tensor* nnl2_ad_get_grad(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_get_grad, ad_tensor is NULL", NULL);
	#endif
	
	if(!ad_tensor->requires_grad) {
		if(ad_tensor->name != NULL) {
			NNL2_ERROR("For tensor %x (namely %s), an attempt was made to obtain gradients when it did not require them. Did you forget`:requires-grad t` ?", ad_tensor, ad_tensor->name);
		} else {
			NNL2_ERROR("For tensor %x, an attempt was made to obtain gradients when it did not require them. Did you forget`:requires-grad t` ?", ad_tensor);
		}
		
		return NULL;
	}
	
	if(ad_tensor->grad_initialized) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
			NNL2_FUNC_EXIT();
		#endif 
		
		return ad_tensor->grad;
	} else {
		NNL2_ERROR("An attempt was made to obtain an uninitialized gradient. Before getting the gradient, first perform backpropagation");
		return NULL;
	}
}

#endif /** NNL2_AD_ACCESSORS_H **/
