#ifndef NNL2_TREFW_GETTER_H
#define NNL2_TREFW_GETTER_H

/** @brief
 * Gets a pointer to a tensor element by coordinates (indexes)
 *
 ** @param tensor
 * Tensor to take element from
 *
 ** @param coords
 * Indices to get element
 *
 ** @param coords_len
 * Length of indices 
 *
 ** @return void* 
 * Pointer to the requested tensor element
 */
void* nnl2_get_raw_tensor_elem(Tensor* tensor, int32_t* coords, int32_t coords_len) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (coords_len != tensor->rank) {
			NNL2_ERROR("Length of indexes (%d) doesn't match tensor dimension (%d)", coords_len, tensor->rank);
			return NULL;
		}
		
		for (int i = 0; i < coords_len; i++) {
			if (coords[i] < 0 || coords[i] >= tensor->shape[i]) {
				NNL2_ERROR("Index at %d is out of bounds (tensor shape at %d is %d)", coords[i], i, tensor->shape[i]);
				return NULL;
			}
		}
	#endif
	
			
	size_t offset = 0;
	for (int i = 0; i < coords_len; i++) {
		offset += coords[i] * tensor->strides[i];
	}
	
	char* elem = (char*)tensor->data + offset * get_dtype_size(tensor->dtype);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
    
    return elem;
}

#endif /** NNL2_TREFW_GETTER_H **/
