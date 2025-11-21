#ifndef NNL2_NARROW_H
#define NNL2_NARROW_H

#include <inttypes.h> // PRIu8

// NNL2

/** @file nnl2_narrow.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contaions nnl2_narrow function
 **/

/** @brief 
 * Creates a narrowed view of the input tensor along the specified dimension
 * 
 ** @param tensor 
 * Input tensor to narrow 
 * 
 ** @param dim 
 * Dimension along which to narrow (0 <= dim < tensor -> rank)
 *
 ** @param start 
 * Starting index in the specified dimension
 *
 ** @param length 
 * Number of elements to include from the start index
 *
 ** @return nnl2_tensor*
 * New tensor view pointing to the narrowed data, or NULL on error
 */
nnl2_tensor* nnl2_narrow(nnl2_tensor* tensor, uint8_t dim, int64_t start, int64_t length) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_narrow, tensor is NULL", NULL);

		if(dim >= tensor -> rank) {
			NNL2_ERROR("In narrow, passed dimension is too big. Dim: %" PRIu8 ", tensor rank: %d. Valid dimensions: 0 to %d", 
						dim, tensor -> rank, tensor -> rank - 1);
						
			return NULL;
		}
		
		if(start >= tensor->shape[dim]) {
			NNL2_ERROR("In narrow, start index %ld is out of bounds for dimension %d of size %d", 
						start, dim, tensor->shape[dim]);
						
		    return NULL;
		}
						
		if(start + length > tensor->shape[dim]) {
			NNL2_ERROR("In narrow, length %ld is invalid for start %ld and dimension size %d", 
						length, start, tensor->shape[dim]);	

		    return NULL;
		}
	#endif
	
	// Create new shape array
	int32_t* new_shape = (int32_t *)malloc(tensor -> rank * sizeof(int32_t));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!new_shape) {
			NNL2_MALLOC_ERROR();
		}
	#endif

    // Copy original shape, replacing the narrowed dimension with the new length
	for(int32_t i = 0; i < tensor->rank; i++) 
		new_shape[i] = (i == dim ? length : tensor -> shape[i]);
	
	// Create empty tensor with the new shape (same dtype and rank)
	nnl2_tensor* result = nnl2_empty(new_shape, tensor -> rank, tensor -> dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("Uninitialized data (empty)");
		}
	#endif
	
	free(new_shape);
	
	// Set result data pointer to the start position in original tensor data
	// start index * stride for the narrowed dimension
	result -> data = tensor -> data + start * tensor -> strides[dim];
	
	// To indicate shared data ownership
	result -> is_view = true;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif 
	
	return result;
}

#endif /** NNL2_NARROW_H **/
