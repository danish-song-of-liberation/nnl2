#ifndef NNL2_HUGE_H
#define NNL2_HUGE_H

/** @brief 
 * Prints a compact representation of a tensor with its metadata
 *
 ** @param tensor
 * Pointer to the tensor to be printed (any rank)
 *
 ** @note
 * Output includes tensor data type and shape information
 * Does not print actual tensor data values, only metadata
 *
 ** @note
 * In safety mode, performs validation of input parameters and tensor structure
 * Handles tensors of any rank including scalars (rank 0)
 *
 ** @example
 * // Print tensor metadata
 * nnl2_print_huge_tensor(my_tensor);
 *
 * // Output format: #<NNL2:TENSOR/FLOAT32 [3x4x5]>
 *
 ** @see get_tensortype_name
 **/
void nnl2_print_huge_tensor(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Comprehensive input validation in safety mode
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor == NULL) {
			NNL2_ERROR("Provided tensor is NULL");
			return;
		}
		
		if (tensor->rank <= 0) {
            NNL2_ERROR("Invalid tensor rank: %d", tensor->rank);
            return;
        }
		
		if (tensor->rank > 0 && tensor->shape == NULL) {
            NNL2_ERROR("Tensor shape array is NULL");
            return;
        }
	#endif
	
	// Print tensor header with library identifier
	printf("#<NNL2:TENSOR/");
	
	TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
	
	printf("%s [", type_name);
	
    // Format shape dimensions as "dim1xdim2xdim3..."
    printf("%d", tensor->shape[0]);
    for (int i = 1; i < tensor->rank; i++) {
        printf("x%d", tensor->shape[i]);
    }
	
	// Close tensor output format
    printf("]>");
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_HUGE_H **/
