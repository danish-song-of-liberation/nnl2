#ifndef NNL2_NARROW_H
#define NNL2_NARROW_H

#include <inttypes.h>  // PRIu8
#include <string.h>
 
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
		
		if(tensor->magic_number != TENSOR_MAGIC_ALIVE) {
            NNL2_ERROR("In narrow, tensor has invalid magic number");
            return NULL;
        }

		if(dim >= tensor -> rank) {
			NNL2_ERROR("In narrow, passed dimension is too big. Dim: %" PRIu8 ", tensor rank: %d. Valid dimensions: 0 to %d", 
						dim, tensor -> rank, tensor -> rank - 1);
						
			return NULL;
		}
		
		if(start >= tensor -> shape[dim]) {
			NNL2_ERROR("In narrow, start index %ld is out of bounds for dimension %d of size %d", 
						start, dim, tensor -> shape[dim]);
						
		    return NULL;
		}
						
		if(start + length > tensor -> shape[dim]) {
			NNL2_ERROR("In narrow, length %ld is invalid for start %ld and dimension size %d", 
						length, start, tensor -> shape[dim]);	

		    return NULL;
		}
	#endif
	
	nnl2_tensor* result = (nnl2_tensor*)malloc(sizeof(nnl2_tensor));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!result) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	// Copy metadata
	result -> ts_type = tensor->ts_type;  
    result -> dtype = tensor->dtype;     
    result -> rank = tensor->rank;      
    result -> magic_number = TENSOR_MAGIC_ALIVE;
    result -> is_view = true; 	 // This is a view!
	 
	result -> shape = (int32_t *)malloc(tensor -> rank * sizeof(int32_t));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!result -> shape) {
			free(result);
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif

    // Copy original shape, replacing the narrowed dimension with new length
    for (int32_t i = 0; i < tensor -> rank; i++) 
        result -> shape[i] = (i == dim) ? length : tensor -> shape[i];
	
	// Allocate and copy strides array
    result -> strides = (int32_t *)malloc(tensor -> rank * sizeof(int32_t));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result -> strides) {
			free(result -> shape);
			free(result);
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	if(!memcpy(result -> strides, tensor -> strides, tensor -> rank * sizeof(int32_t))) {
		NNL2_ERROR("In narrow, failed to memcpy");
		free(result->strides);
		free(result->shape);
		free(result);
		return NULL;
	}
	
	// Calculate data pointer offset
    size_t element_size = get_dtype_size(tensor -> dtype);
    result -> data = (char*)tensor -> data + start * tensor -> strides[dim] * element_size;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Created narrow view: dim=%d, start=%ld, length=%ld", dim, start, length);
        NNL2_FUNC_EXIT();
    #endif 
	
	return result;
}

#endif /** NNL2_NARROW_H **/
