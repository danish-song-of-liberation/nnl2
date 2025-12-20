#ifndef NNL2_ASSIGN_ROW_H
#define NNL2_ASSIGN_ROW_H 

// NNL2

void nnl2_naive_assign_row(nnl2_tensor* dst, int seq_index, nnl2_tensor* src) {
    size_t elem_size = get_dtype_size(dst->dtype);
	
	for(int b = 0; b < dst->shape[0]; b++) {
		char* dst_ptr = (char*)dst->data;
		dst_ptr += b * dst->strides[0] * elem_size;
		dst_ptr += seq_index * dst->strides[1] * elem_size;
		
		char* src_ptr = (char*)src->data;
		src_ptr += b * src->strides[0] * elem_size; 
		
		size_t bytes_to_copy = dst->shape[2] * elem_size;
		memcpy(dst_ptr, src_ptr, bytes_to_copy);
	}
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for assign_row operation
 * @details
 * Array follows the common backend registration pattern for assign_row
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for row assignment
 * 
 * @see nnl2_naive
 * @see naive_assign_row
 */
Implementation assign_row_backends[] = {
    REGISTER_BACKEND(nnl2_naive_assign_row, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for assign_row operation
 * @ingroup backend_system
 */
assignrowfn nnl2_assign_row;

/** 
 * @brief Sets the backend for assign_row operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for assign_row
 * @see SET_BACKEND_BY_NAME
 * @see assign_row_backends
 */
void set_assign_row_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(assign_row_backends, nnl2_assign_row, backend_name);
}

#endif /** NNL2_ASSIGN_ROW_H **/
