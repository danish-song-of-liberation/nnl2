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

/** @brief
 * Creates a new tensor of the same shape and type, with random values from a normal distribution
 *
 ** @param tensor
 * The initial tensor from which shape, rank and dtype are taken
 *
 ** @param from
 * A pointer to the lower bound of the random range (optional, can be NULL)
 *
 ** @param to
 * A pointer to the upper bound of the random range (optional, can be NULL)
 *
 ** @return 
 * Tensor* Pointer to a new tensor filled with random values from normal distribution
 * 
 ** @note
 * If 'from' and 'to' are NULL, uses default range for the dtype
 * The range parameters must match the tensor's dtype
 *
 ** @warning
 * It is necessary to ensure that the range parameters match the tensor's dtype
 *
 ** @see nnl2_randn
 **/
Tensor* nnl2_uniform_like(Tensor* tensor, void* from, void* to) {
	return uniform(tensor->shape, tensor->rank, tensor->dtype, from, to);
}

/** @brief
 * Creates a new tensor of the same shape and type, initialized with Xavier/Glorot initialization
 *
 ** @param tensor
 * The initial tensor from which shape, rank and dtype are taken
 *
 ** @param in
 * Number of input units (fan_in) for the layer
 *
 ** @param out
 * Number of output units (fan_out) for the layer
 *
 ** @param gain
 * Scaling factor for the initialization (optional, typically 1.0)
 *
 ** @param dist
 * 6 for uniform, 2 for normal distribution
 *
 ** @return nnl2_tensor* 
 * Pointer to a new tensor filled with Xavier-initialized values
 *
 ** @see xavier
 **/
nnl2_tensor* nnl2_xavier_like(nnl2_tensor* tensor, int in, int out, float gain, float dist) {
	return xavier(tensor->shape, tensor->rank, tensor->dtype, in, out, gain, dist);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with uniform random values [0, 1)
 *
 ** @param tensor
 * The initial tensor from which shape, rank and dtype are taken
 *
 ** @return nnl2_tensor*
 * Pointer to a new tensor filled with uniform random values [0, 1)
 *
 ** @example
 * // Create a tensor like existing one with uniform random values
 * nnl2_tensor* original = nnl2_ones((int[]){3, 4}, 2, FLOAT32);
 * nnl2_tensor* random_copy = nnl2_rand_like(original);
 *
 ** @see nnl2_rand
 ** @see nnl2_randn_like
 **/
nnl2_tensor* nnl2_rand_like(nnl2_tensor* tensor) {
	return nnl2_rand(tensor -> shape, tensor -> rank, tensor -> dtype);
}

/** @brief
 * Creates a new tensor of the same shape and type, filled with random values from normal distribution N(mean, std^2)
 *
 ** @param tensor
 * The initial tensor from which shape, rank and dtype are taken
 *
 ** @param mean
 * Mean of the normal distribution
 *
 ** @param std
 * Standard deviation of the normal distribution
 *
 ** @return nnl2_tensor*
 * Pointer to a new tensor filled with random values from N(mean, std^2)
 *
 ** @example
 * // Create a tensor like existing one with standard normal values
 * nnl2_tensor* original = nnl2_zeros((int[]){2, 2}, 2, FLOAT64);
 * nnl2_tensor* normal_copy = nnl2_randn_like(original, 0.0, 1.0);
 *
 ** @example
 * // Create a tensor like existing one with custom normal distribution
 * nnl2_tensor* weights = nnl2_empty((int[]){128, 64}, 2, FLOAT32);
 * nnl2_tensor* initialized = nnl2_randn_like(weights, 0.0, 0.02);
 *
 ** @see nnl2_randn
 ** @see nnl2_rand_like
 **/
nnl2_tensor* nnl2_randn_like(nnl2_tensor* tensor, double mean, double std) {
	return nnl2_randn(tensor -> shape, tensor -> rank, tensor -> dtype, mean, std);
}

#endif /** NNL2_LIKE_CONSTRUCTORS_H **/
