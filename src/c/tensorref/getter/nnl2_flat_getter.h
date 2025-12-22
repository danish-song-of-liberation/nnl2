#ifndef NNL2_FLAT_GETTER_H
#define NNL2_FLAT_GETTER_H

/** @brief
 * Gets a pointer to a tensor element by linear index
 *
 ** @param tensor 
 * Tensor to take element with linear index
 *
 ** @param at
 * Linear index
 *
 ** @return void* 
 * Pointer to the requested tensor element
 */
void* nnl2_get_raw_tensor_elem_at(Tensor* tensor, size_t at) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
		if(at >= total_elems) {
			NNL2_ERROR("Index out of bounds: index %zd exceeds tensor size %zd", at, total_elems);
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
	return ((char*)tensor->data + at * get_dtype_size(tensor->dtype));
}

#endif /** NNL2_FLAT_GETTER_H **/
