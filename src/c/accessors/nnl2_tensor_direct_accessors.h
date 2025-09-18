#ifndef NNL2_TENSOR_DIRECT_ACCESSORS_H
#define NNL2_TENSOR_DIRECT_ACCESSORS_H

#include "../nnl2_tensor_backend.h"

/// NNL2

/** @file nnl2_tensor_direct_accessors.h
 ** @brief Contains direct accessors like get_tensor_dtype, etc.
 ** @copyright MIT License
 ** @date 2025
 **
 ** Filepath: nnl2/src/c/accessors/nnl2_tensor_direct_accessors.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2		
 **/ 

/** @brief
 * Get the raw data pointer from a tensor
 *
 ** @param tensor
 * Pointer to the tensor
 *
 ** @return
 * Void pointer to the tensor's underlying data storage
 */
void* nnl2_get_tensor_data(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID_RET_PNTR);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->data;
}

/** @brief 
 * Get the pointer to the tensor's strides array
 *
 ** @param tensor 
 * Input tensor
 *
 ** @return 
 * Pointer to the tensor's strides array
 */
int32_t* nnl2_get_tensor_strides(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID_RET_PNTR);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->strides;
}

/** @brief 
 * Get the shape array of a tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return 
 * Pointer to the shape array containing dimension sizes
 *
 ** @note 
 * The returned array has length equal to the tensor's rank
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @warning 
 * Modifying the returned array may corrupt the tensor structure
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * int32_t* shape = nnl2_get_tensor_shape(t);
 * printf("Tensor shape: [");
 * for (int i = 0; i < nnl2_get_tensor_rank(t); i++) {
 *     printf("%d%s", shape[i], i < get_tensor_rank(t)-1 ? ", " : "");
 * }
 * printf("]\n");
 */
int32_t* nnl2_get_tensor_shape(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID_RET_PNTR);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->shape;
}

/** @brief 
 * Get the data type of a tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return TensorType 
 * Data type of the tensor elements
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @note Returns 
 * NNL2_TENSOR_TYPE_INVALID if tensor is invalid in safety mode
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * TensorType dtype = nnl2_get_tensor_dtype(t);
 * if (dtype == FLOAT32) {
 *     printf("Tensor is float32 type\n");
 * }
 */
TensorType nnl2_get_tensor_dtype(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID)
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->dtype;
}

/** @brief
 * Get rank (ndim) of  tensor
 *
 ** @param tensor 
 * Pointer to the tensor structure
 *
 ** @return 
 * The rank of the tensor (number of dimensions)
 *
 ** @note 
 * In safety mode, validates the tensor pointer before access
 *
 ** @note 
 * Returns NNL2_TENSOR_TYPE_INVALID (-1) if tensor is invalid in safety mode
 *
 * @example
 * Tensor* t = nnl2_zeros(...);
 * int32_t rank = nnl2_get_tensor_rank(t);
 * printf("Tensor rank: %d\n", rank);
 */
int32_t nnl2_get_tensor_rank(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_TENSOR(tensor, NNL2_TENSOR_TYPE_INVALID)
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor->rank;
}

/** @brief 
 * Get the dimension size at the specified index of a tensor's shape
 * 
 ** @param tensor 
 * Pointer to the tensor
 *
 ** @param index 
 * Index of the dimension to retrieve 
 *
 ** @return int32_t
 * The size of the dimension at the specified index
 */
int32_t nnl2_shape_at(Tensor* tensor, int32_t index) {
	return tensor->shape[index];
}

#endif /** NNL2_TENSOR_DIRECT_ACCESSORS_H **/