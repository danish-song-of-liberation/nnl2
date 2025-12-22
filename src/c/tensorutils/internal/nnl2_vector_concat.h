#ifndef NNL2_VECTOR_CONCAT_H
#define NNL2_VECTOR_CONCAT_H

/** @brief
 * Concatenates multiple tensors along a new dimension to form a single vector
 *
 ** @param tensors
 * Array of pointers to input tensors to be concatenated
 *
 ** @param count
 * Number of tensors in the array
 *
 ** @param dtype
 * Data type for the resulting tensor
 *
 ** @return nnl2_tensor*
 * Pointer to a new tensor containing the concatenated result
 */
nnl2_tensor* nnl2_naive_vector_concat(nnl2_tensor** tensors, size_t count, nnl2_tensor_type dtype) {
	size_t numel = 0;
	
	for(size_t it = 0; it < count; it++) numel += nnl2_product(tensors[it] -> shape, tensors[it] -> rank);

	nnl2_tensor* tensor = nnl2_empty((int[]){ numel }, 1, dtype);
	
	size_t offset = 0;
    for (size_t it = 0; it < count; it++) {
        size_t elem_count = nnl2_product(tensors[it]->shape, tensors[it]->rank);
        size_t byte_size = elem_count * get_dtype_size(dtype);
        memcpy((char*)tensor->data + offset, tensors[it]->data, byte_size);
        offset += byte_size;
    }
	
	return tensor;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for vector concatenation operation
 * @details
 * Array follows the common backend registration pattern for concatenation
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for vector concatenation
 * 
 * @see nnl2_naive
 * @see naive_vector_concat
 */
Implementation vector_concat_backends[] = {
    REGISTER_BACKEND(nnl2_naive_vector_concat, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for vector concatenation operation
 * @ingroup backend_system
 */
vectorconcatfn nnl2_vector_concat;

/** 
 * @brief Sets the backend for vector concatenation operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for concatenation
 * @see SET_BACKEND_BY_NAME
 * @see vector_concat_backends
 */
void set_vector_concat_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(vector_concat_backends, nnl2_vector_concat, backend_name);
}

#endif /** NNL2_VECTOR_CONCAT_H **/
