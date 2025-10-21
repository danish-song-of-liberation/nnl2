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

#endif /** NNL2_AD_ACCESSORS_H **/
