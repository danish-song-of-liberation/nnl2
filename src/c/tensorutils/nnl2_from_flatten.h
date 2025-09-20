#ifndef NNL2_FROM_FLATTEN_H
#define NNL2_FROM_FLATTEN_H

/** @brief
 * Creates a tensor from a flattened array
 *
 ** @param arr
 * Pointer to the flattened array data
 *
 ** @param num_elems_arr
 * Number of elements in the array
 *
 ** @param shape
 * Shape of the resulting tensor
 *
 ** @param rank
 * Rank of the resulting tensor
 *
 ** @param dtype
 * Data type of the resulting tensor
 *
 ** @return
 * New tensor with data copied from the array
 */
Tensor* make_tensor_from_flatten(void* arr, size_t num_elems_arr, int* shape, int rank, TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t num_elems_tensor = product(shape, rank);
	
	if(num_elems_tensor != num_elems_arr) {
		NNL2_ERROR("The number of elements in the specified array does not match the specified shapes");
		return NULL;
	}
	
	Tensor* result = nnl2_empty(shape, rank, dtype);
	fill_tensor_with_data(result, arr, num_elems_tensor);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#endif /** NNL2_FROM_FLATTEN_H **/