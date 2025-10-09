#ifndef NNL2_FOREIGN_SHAPE
#define NNL2_FOREIGN_SHAPE

/** @brief 
 * Returns the number of rows (the first dimension) of a tensor
 * 
 ** @param tensor
 * Input tensor
 *
 ** @return int32_t
 * Number of rows in a tensor
 */
int32_t nnl2_nrows (Tensor* tensor) { 
	return tensor->shape[0];
}

/** @brief 
 * Returns the number of cols (the second dimension) of a tensor
 * 
 ** @param tensor
 * Input tensor
 *
 ** @return int32_t
 * Number of cols in a tensor
 */
int32_t nnl2_ncols (Tensor* tensor) {
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(tensor->rank < 2) {
			NNL2_ERROR("Can't extract number of cols in passed tensor (tensor rank is less than 2)");
			return -1;
		}
	#endif
	
	return tensor->shape[1];
}

#endif /** NNL2_FOREIGN_SHAPE **/
