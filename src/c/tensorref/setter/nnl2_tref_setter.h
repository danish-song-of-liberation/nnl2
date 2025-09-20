#ifndef NNL2_TREF_SETTER_H
#define NNL2_TREF_SETTER_H

/** @brief
 * Sets a subtensor or single element in the destination tensor
 *
 ** @param dest
 * Destination tensor to modify
 *
 ** @param dest_shape
 * Array specifying the starting indices for each dimension
 *
 ** @param dest_rank
 * Number of dimensions specified in dest_shape
 *
 ** @param src
 * Source tensor to copy from
 *
 ** @param src_shape
 * Array specifying which indices to copy from source
 *
 ** @param src_rank
 * Number of dimensions specified in src_shape
 */
void nnl2_tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank);

/** @brief
 * Naive tensor reference setter for assignment operations
 *
 * Handles assignment of both scalar values and subtensors to a target tensor
 * Supports wildcard indexing (-1) for iterating through dimensions
 * In lisp wrapper, all '* are automatically converted to -1
 *
 ** @param tensor
 * Target tensor to modify
 *
 ** @param shape
 * Array of indices specifying the target location
 *
 ** @param rank
 * Number of dimensions specified in shape array
 *
 ** @param change_with
 * Pointer to the data to assign (scalar or tensor)
 *
 ** @param is_tensor
 * Flag indicating if change_with is a tensor (true) or scalar (false)
 */
void nnl2_naive_tref_setter(Tensor* tensor, int* shape, int rank, void* change_with, bool is_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    TensorType tensor_dtype = tensor->dtype;

	// Handle tensor assignment (subtensor to subtensor)
    if (is_tensor) {
        Tensor* sub_tensor = (Tensor*)change_with;

		// Additional checks depending on the security level
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
			if (tensor->rank - rank != sub_tensor->rank) {
				NNL2_ERROR("Rank mismatch in tensor assignment");
				return;
			}
			
			if (tensor->dtype != sub_tensor->dtype) {
				NNL2_ERROR("Data type mismatch in tensor assignment");
				return;
			}
		#endif
        
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Validate shape compatibility for all dimensionss
			for (int i = 0; i < sub_tensor->rank; i++) {
				if (tensor->shape[rank + i] != sub_tensor->shape[i]) {
					NNL2_ERROR("Shape mismatch in dimension %d", i);
					return;
				}
			}
		#endif
        
        int* full_dest_shape = malloc(sizeof(int) * tensor->rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!full_dest_shape) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif
        
        memcpy(full_dest_shape, shape, sizeof(int) * rank);
        
        for (int i = rank; i < tensor->rank; i++) {
            full_dest_shape[i] = -1;
        }
        
        int* src_iter_shape = malloc(sizeof(int) * sub_tensor->rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!src_iter_shape) {
				free(full_dest_shape);
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif
        
        for (int i = 0; i < sub_tensor->rank; i++) {
            src_iter_shape[i] = -1;
        }
        
        nnl2_tensor_set_subtensor(tensor, full_dest_shape, rank, sub_tensor, src_iter_shape, 0);
        
        free(src_iter_shape);
        free(full_dest_shape);
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		
        return;
    }
    
	// Handle wildcard indices by recursively iterating through those dimensions
    for(int i = 0; i < rank; i++) {
        if(shape[i] == -1) {
            for(int shape_i = 0; shape_i < tensor->shape[i]; shape_i++) {
                shape[i] = shape_i;
                nnl2_naive_tref_setter(tensor, shape, rank, change_with, is_tensor);
            }
            
            shape[i] = -1;
            return;
        }
    }
    
    if(rank == tensor->rank) {
		// Handling case when reached target element for scalar assignment
        switch(tensor_dtype) {
            case FLOAT64: {
                double* change_elem = (double*)change_with;
                double* elem = (double*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            case FLOAT32: {
                float* change_elem = (float*)change_with;
                float* elem = (float*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            case INT32: {
                int32_t* change_elem = (int32_t*)change_with;
                int32_t* elem = (int32_t*)nnl2_view(tensor, shape, rank);
                *elem = *change_elem;
                break;
            }
			
            default: {
                NNL2_TYPE_ERROR(tensor_dtype);			
                return;
            }
        }
    } else {
		// Handling case when need to go deeper into the tensor hierarchy
        int new_rank = rank + 1;
        int* subshape = malloc(sizeof(int) * new_rank);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!subshape) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
		#endif

        memcpy(subshape, shape, sizeof(int) * rank);

        for (int i = 0; i < tensor->shape[rank]; i++) {
            subshape[rank] = i;
            nnl2_naive_tref_setter(tensor, subshape, new_rank, change_with, is_tensor);
        }

        free(subshape);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief
 * Recursively copies data from source tensor to destination subtensor
 * See docs at declaration 
 *
 ** @see nnl2_tensor_set_subtensor
 **/
void nnl2_tensor_set_subtensor(Tensor* dest, int* dest_shape, int dest_rank, Tensor* src, int* src_shape, int src_rank) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
    if (src_rank == src->rank) {
        void* dest_ptr = nnl2_view(dest, dest_shape, dest_rank);
        void* src_ptr = nnl2_view(src, src_shape, src_rank);
        
		// Determine data type size for memcpy
        size_t type_size;
        switch (dest->dtype) {
            case FLOAT64: type_size = sizeof(double); break;
            case FLOAT32: type_size = sizeof(float); break;
            case INT32: type_size = sizeof(int32_t); break;
            default: NNL2_TYPE_ERROR(dest->dtype); return;
        }
        
		// Perform the actual data copy
        memcpy(dest_ptr, src_ptr, type_size);
        return;
    }

    // Handle wildcard indices in source by iterating through that dimension
    if (src_shape[src_rank] == -1) {
        for (int i = 0; i < src->shape[src_rank]; i++) {
            src_shape[src_rank] = i;
            dest_shape[dest_rank] = i;
            nnl2_tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
        }
		
        src_shape[src_rank] = -1;
        dest_shape[dest_rank] = -1;
    } else {
        dest_shape[dest_rank] = src_shape[src_rank];
        nnl2_tensor_set_subtensor(dest, dest_shape, dest_rank + 1, src, src_shape, src_rank + 1);
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for tensor reference setter operation
 * @details
 * Array follows the common backend registration pattern for tensor reference setting operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive_tref_setter
 * @see nnl2_naive
 */
Implementation tref_setter_backends[] = {
	REGISTER_BACKEND(nnl2_naive_tref_setter, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for tensor reference setter operation
 * @ingroup backend_system 
 */
trefsetterfn tref_setter;

/** 
 * @brief Sets the backend for tensor reference setter operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 */
void set_tref_setter_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(tref_setter_backends, tref_setter, backend_name);
}

#endif /** NNL2_TREF_SETTER_H **/
