#ifndef NNL2_FREE_H
#define NNL2_FREE_H

/** @brief
 * Frees the memory allocated for the tensor.
 *
 ** @param tensor 
 * A pointer to a tensor whose memory needs to be freed. If tensor is null, the function does nothing.
 *
 ** @note
 * After calling this function tensor pointer becomes invalid. 
 * Do not attempt to access the tensor after it has been freed (although you'll try it anyway, you idiot).
 *
 ** @note 
 * Additional checks are added depending on the safety level
 *
 ** @see FREE_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 **/
void nnl2_free_tensor(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if (tensor == NULL) {
			NNL2_ERROR("Invalid data (NULL) was passed to the tensor release");
			return;
		}
	#endif
	
	// Additional checks
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if (tensor->rank < 0) {
			NNL2_ERROR("Invalid tensor rank: %d", tensor->rank);
		}
		
		if (tensor->shape == NULL && tensor->rank > 0) {
			NNL2_ERROR("NULL shape array with rank %d", tensor->rank);
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_DEBUG("Freeing tensor [%p] (rank: %d)", (void*)tensor, tensor->rank);
    #endif
	
	#if NNL2_SAFETY_MODE <= NNL2_SAFETY_MODE_MODERATE
		free(tensor->shape); 
		free(tensor->strides);
		FREE_ALIGNED(tensor->data);
	#else
		// Safe freeing
		if (tensor->shape != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing shape array [%p]", (void*)tensor->shape);
			#endif
			
			free(tensor->shape);
		}
    
		if (tensor->strides != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing strides [%p]", (void*)tensor->strides);
			#endif
			
			free(tensor->strides);
		}
	
		if (!tensor->is_view && tensor->data != NULL) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_DEBUG("Freeing data [%p]", (void*)tensor->data);
			#endif
			
			FREE_ALIGNED(tensor->data);
		}
	#endif
	
	free(tensor);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_FREE_H **/
