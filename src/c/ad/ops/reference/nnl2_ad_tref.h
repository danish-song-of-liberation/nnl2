#ifndef NNL2_AD_TREF_H
#define NNL2_AD_TREF_H

/** @file nnl2_ad_tref.h
 ** @brief AD implementation for tref_getter operation
 ** @date 2025
 ** @copyright MIT License
 **/

/** @brief 
 * Reverse mode backward pass for tref_getter operation
 *
 ** @param tensor
 * The output tensor from tref_getter operation that needs gradient computation
 *
 ** @details
 * Derivative of tref_getter: propagates gradient back to original tensor using stored indices
 * Handles gradient accumulation for tensor indexing operations that return a copy
 *
 ** @exception NNL2Error
 * If tensor is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->data is NULL and safety mode is MAX, function returns early
 *
 ** @exception NNL2Error
 * If tensor->roots[0] is NULL and safety mode is MAX, function returns early
 *
 ** @see nnl2_ad_tref_getter()
 ** @see nnl2_ad_reverse_derivative_tref_getter()
 **/
static void nnl2_ad_reverse_backward_tref_getter(nnl2_ad_tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_reverse_backward_tref_getter, passed AD tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor -> data, "In function nnl2_ad_reverse_backward_tref_getter, passed AD tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor -> roots[0], "In function nnl2_ad_reverse_backward_tref_getter, root tensor is NULL");
	#endif
	
	int32_t* indices = (int32_t*)tensor -> extra_correspondence;
	
	// Compute the derivative of tref_getter operation
    nnl2_ad_reverse_derivative_tref_getter(tensor, tensor -> roots[0], indices, tensor -> extra_integer, tensor -> extra_bool);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

/** @brief 
 * Create an automatic differentiation tensor for tref_getter operation
 *
 ** @param tensor 
 * Input tensor for tref_getter operation
 *
 ** @param indices 
 * Array of indices for indexing operation
 *
 ** @param num_indices 
 * Number of indices in the indices array
 *
 ** @param ad_mode 
 * Automatic differentiation mode (reverse/p1/p2/p3)
 *
 ** @param track_graph 
 * Whether to track this operation in computation graph
 *
 ** @param force
 * If true, returns raw scalar value instead of AD tensor
 *  
 ** @return void*
 * New AD tensor containing tref_getter result, or scalar value if force=true, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if tensor is NULL (SAFETY_MODE_MODERATE+)
 *
 ** @exception NNL2Error
 * Returns NULL if tensor components are NULL (SAFETY_MODE_FULL+)
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails for result tensor
 *
 ** @exception NNL2Error
 * Returns NULL if tref_getter operation fails on input tensor
 *
 ** @exception NNL2Error
 * Returns NULL if gradient tensor allocation fails
 *
 ** @exception NNL2Error
 * Returns NULL if roots array allocation fails when track_graph=true
 *
 ** @exception NNL2Error
 * Returns NULL if unknown AD mode is specified
 *
 ** @see nnl2_tref_getter()
 ** @see nnl2_empty()
 ** @see nnl2_free_ad_tensor()
 **/
void* nnl2_ad_tref_getter(nnl2_ad_tensor* tensor, int32_t* indices, uint8_t num_indices, nnl2_ad_mode ad_mode, bool track_graph, bool force) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Basic null checks for input tensors
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "In function nnl2_ad_tref_getter, AD tensor is NULL", NULL);
	#endif
	
	// Comprehensive null checks for tensor components
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_FULL
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor -> data, "In function nnl2_ad_tref_getter, AD tensor data is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor -> data -> shape, "In function nnl2_ad_tref_getter, AD tensor shape is NULL", NULL);
	#endif
	
	bool scalar_exit_p = tensor -> data -> rank == num_indices;
	
    if(force && scalar_exit_p) {
        void* scalar_result = (void*)nnl2_tref_getter(tensor->data, indices, num_indices);
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return scalar_result;
		
	} else {
		nnl2_ad_tensor* result = malloc(sizeof(nnl2_ad_tensor));
		if(!result) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
		
		// To protect against re-freeing
		result -> magic_number = TENSOR_MAGIC_ALIVE;
		
		// Compute indexing operation (returns a copy)
		if(!scalar_exit_p) {
			result -> data = nnl2_tref_getter(tensor -> data, indices, num_indices);
			if(!result->data) {
				NNL2_ERROR("In function nnl2_ad_tref_getter, failed to allocate tref_getter tensor");
				free(result);
				return NULL;
			}
			
		} else {
			result -> data = nnl2_empty((int32_t[]){1}, 1, tensor -> data -> dtype);
			result -> data -> data = (void*)nnl2_tref_getter(tensor -> data, indices, num_indices);
		}
		
		// Allocate gradient tensor with same shape as result
		result -> grad = nnl2_empty(result -> data -> shape, result -> data -> rank, result -> data -> dtype);
		if(!result -> grad) {
			NNL2_ERROR("In function nnl2_ad_tref_getter, failed to allocate gradient tensor");
			nnl2_free_tensor(result->data);
			free(result);
			return NULL;
		}
		
		// Build computational graph if tracking is enabled
		if(track_graph) {
			result -> num_roots = 1;
			result -> roots = (nnl2_ad_tensor**)malloc(sizeof(*result->roots));
			if(!result -> roots) {
				NNL2_MALLOC_ERROR();
				nnl2_free_tensor(result -> data);
				nnl2_free_tensor(result -> grad);
				free(result);
				return NULL;
			}
		
			result -> roots[0] = tensor;
			
			// Set the appropriate backward function based on AD mode
			switch(ad_mode) {
				case nnl2_ad_reverse_mode: {
					result -> backward_fn = nnl2_ad_reverse_backward_tref_getter;  
					break;
				}
				
				default: {
					NNL2_UNKNOWN_AD_MODE_ERROR(ad_mode);
					nnl2_free_ad_tensor(result);
					return NULL;
				}
			}
		} else {
			// No computational graph tracking
			result -> num_roots = 0;
			result -> roots = NULL;
			result -> backward_fn = NULL;
		}
		
		// Initialize tensor metadata
		result -> requires_grad = tensor -> requires_grad;
		result -> grad_initialized = false;
		result -> is_leaf = false;
		result -> extra_multiplier = 1.0f;
		result -> extra_bool = scalar_exit_p;
		result -> extra_correspondence = (int32_t*)indices;
		result -> name = NULL;
		result -> ts_type = nnl2_type_ad;
		result -> visited_gen = 0;
		result -> extra_integer = num_indices;
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		
		return result;
	}
}

#endif /** NNL2_AD_TREF_H **/
