#ifndef NNL2_TREFW_SETTER_H
#define NNL2_TREFW_SETTER_H

/** @brief
 * Sets a tensor element by coordinates (indexes)
 *
 ** @param tensor
 * Tensor to set element in
 *
 ** @param coords
 * Indices to set element at
 *
 ** @param coords_len
 * Length of indices
 *
 ** @param with
 * Pointer to data to set
 */
void nnl2_set_raw_tensor_elem(Tensor* tensor, int32_t* coords, int32_t coords_len, void* with) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (coords_len != tensor->rank) {
            NNL2_ERROR("Length of indexes (%d) doesn't match tensor dimension (%d)", coords_len, tensor->rank);
            return;
        }
        
        for (int i = 0; i < coords_len; i++) {
            if (coords[i] < 0 || coords[i] >= tensor->shape[i]) {
                NNL2_ERROR("Index at %d is out of bounds (tensor shape at %d is %d)", coords[i], i, tensor->shape[i]);
                return;
            }
        }
    #endif
    
    size_t offset = 0;
    for (int i = 0; i < coords_len; i++) {
        offset += coords[i] * tensor->strides[i];
    }
    
    size_t elem_size = get_dtype_size(tensor->dtype);
    char* dest = (char*)tensor->data + offset * elem_size;
    memcpy(dest, with, elem_size);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_TREFW_SETTER_H **/
