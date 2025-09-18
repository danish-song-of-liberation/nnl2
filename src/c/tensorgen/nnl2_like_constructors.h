#ifndef NNL2_LIKE_CONSTRUCTORS_H
#define NNL2_LIKE_CONSTRUCTORS_H

/** @brief
 * Creates a new tensor of the same shape and type, but with uninitialized data
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* Pointer to a new tensor with uninitialized data
 *
 ** @note
 * The data in the returned tensor is not initialized and may contain garbage
 *
 ** @see nnl2_empty
 **/
Tensor* nnl2_empty_like(Tensor* tensor) {
	return nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with zeros
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* A pointer to a new tensor filled with zeros
 *
 ** @note
 * All elements of the returned tensor are initialized with zero values
 *
 ** @see nnl2_zeros
 **/
Tensor* nnl2_zeros_like(Tensor* tensor) {
	return nnl2_zeros(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief 
 * Creates a new tensor of the same shape and type, filled with units
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @return
 * Tensor* A pointer to a new tensor filled with ones
 *
 ** @note 
 * All elements of the returned tensor are initialized with ones
 *
 ** @see nnl2_ones
 **/
Tensor* nnl2_ones_like(Tensor* tensor) {
	return nnl2_ones(tensor->shape, tensor->rank, tensor->dtype);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with the specified value
 *
 ** @param tensor
 * The initial tensor from which a new tensor is created
 *
 ** @param filler
 * A pointer to the value that will be filled into the tensor
 *
 ** @return 
 * Tensor* Pointer to a new tensor filled with the specified value
 * 
 ** @note
 * The filler value must be of the correct dtype tensor type
 *
 ** @warning
 * It is necessary to ensure that the filler value and tensor type match
 *
 ** @see nnl2_full
 **/
Tensor* nnl2_full_like(Tensor* tensor, void* filler) {
	return nnl2_full(tensor->shape, tensor->rank, tensor->dtype, filler);
}

#endif /** NNL2_LIKE_CONSTRUCTORS_H **/
