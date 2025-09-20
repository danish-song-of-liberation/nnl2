#ifndef NNL2_CAST_H
#define NNL2_CAST_H

/** @brief
 * The function is exclusively for the lisp wrapper, 
 * and in C, use nnl2_copy (the same argument list as here)
 *
 ** @param tensor
 * Input tensor
 *
 ** @param cast_to
 * Specifies which type of tensor to cast
 *
 ** @return
 * Tensor with the same data but a new (specified) type
 */
Tensor* nnl2_cast(Tensor* tensor, TensorType cast_to) {
	return nnl2_copy(tensor, cast_to);
}

#endif /** NNL2_CAST_H **/
