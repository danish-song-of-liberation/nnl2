#ifndef NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H
#define NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H

/** @file nnl2_ad_backpropagation_through_time.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Automatic differentiation backpropagation through time implementation
 **/

/** @brief Reduction **/
#define nnl2_ad_bptt(tensor) nnl2_ad_backpropagation_through_time(tensor)

/** @brief 
 * Performs backpropagation through time for recurrent networks
 *
 ** @param tensor 
 * The output tensor from which backpropagation through time starts
 *
 ** @param retain_graph
 * If false, clears the computational graph after backward pass
 *
 ** @warning
 * This is not exactly BPTT in the usual sense. 
 * It is an imitation of BPTT by assigning the first 
 * available leaf tensora to the original tensor.
 * This works for a = a * a and other recursive calculations
 */
void nnl2_ad_backpropagation_through_time(nnl2_ad_tensor* restrict tensor, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_backpropagation_through_time, tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots, "In function nnl2_ad_backpropagation_through_time, tensor roots is NULL");
	#endif
	
    // Find leaf tensor for BPTT gradient accumulation
    nnl2_ad_tensor* leaf_tensor = nnl2_ad_find_leaf(tensor);
	if(!leaf_tensor) {
		NNL2_ERROR("Could not find a leaf tensor in BPTT");
	}
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(leaf_tensor, "Failed to find leaf tensor in backpropagation through time");
	#endif
    
    // Build topological order of computational graph
    int topo_size;
    nnl2_ad_tensor** topo = nnl2_ad_build_topo(tensor, &topo_size);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(topo, "Failed to build topo in backpropagation through time");
	#endif

	void* zero_value = nnl2_get_zero_value(tensor->grad->dtype);
	if (!zero_value) {
		NNL2_ERROR("Failed to allocate zero value for dtype %d", tensor->grad->dtype);
		free(topo);  
		return;
	}

	void* one_value = nnl2_get_one_value(tensor->grad->dtype);
	if (!one_value) {
		NNL2_ERROR("Failed to allocate one value for dtype %d", tensor->grad->dtype);
		free(topo);  
		return;
	}
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!zero_value || !one_value) {
			NNL2_ERROR("Failed to obtain scalar constants for dtype %d", tensor->grad->dtype);
			free(topo);
			return;
		}
	#endif
	
    // Initialize gradients to zero for all tensors that require gradients
    for (int i = 0; i < topo_size; i++) {
        if (topo[i]->requires_grad && !topo[i]->grad_initialized) {
            inplace_fill(topo[i]->grad, zero_value, topo[i]->grad->dtype);
            topo[i]->grad_initialized = true;
        }
    }
    
    // Initialize the output tensor's gradient to 1 to start backpropagation
    bool success = inplace_fill(tensor->grad, one_value, tensor->grad->dtype);
    if (!success) {
		NNL2_ERROR("Failed to initialize passed tensor with 1.0 in backpropagation through time");
        free(topo);
        return;
    }
    
    tensor->grad_initialized = true;

    // Process nodes in reverse topological order
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward_fn) {
			// Execute backward function to compute gradients for this operation
            topo[i]->backward_fn(topo[i]);  
        }
    }
    
    tensor->grad = leaf_tensor->grad;
	
	// Clear computational graph if not needed for future backward passes
	if(!retain_graph) nnl2_ad_clear_graph(topo, topo_size);
    
    // Free topological order array
    free(topo);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_BACKPROPAGATION_THROUGH_TIME_H **/
