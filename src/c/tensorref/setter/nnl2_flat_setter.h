#ifndef NNL2_FLAT_SETTER_H
#define NNL2_FLAT_SETTER_H

/** @brief
 * Sets a tensor element by linear index
 *
 ** @param tensor 
 * Tensor to set element in
 *
 ** @param at
 * Linear index
 *
 ** @param with
 * Pointer to data to set
 */
void nnl2_set_raw_tensor_elem_at(Tensor* tensor, size_t at, void* with) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        size_t total_elems = product(tensor->shape, tensor->rank);
        if(at >= total_elems) {
            NNL2_ERROR("Index out of bounds: index %zd exceeds tensor size %zd", at, total_elems);
            return;
        }
    #endif
	
	size_t elem_size = get_dtype_size(tensor->dtype);
    char* dest = (char*)tensor->data + at * elem_size;
	memcpy(dest, with, elem_size);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_FLAT_SETTER_H **/
