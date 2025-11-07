#ifndef NNL2_AD_BACKPROPAGATION_H
#define NNL2_AD_BACKPROPAGATION_H

/** @file nnl2_ad_backpropagation.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Automatic differentiation backpropagation implementation
 **/

/** @brief 
 * Performs backpropagation through the computational graph
 *
 ** @param tensor 
 * The output tensor from which backpropagation starts (usually loss) 
 *
 ** @param retain_graph
 * If false, clears the computational graph after backward pass
 */
void nnl2_ad_backpropagation(nnl2_ad_tensor* tensor, bool retain_graph) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_ad_backpropagation, tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->roots, "In function nnl2_ad_backpropagation, tensor roots is NULL");
	#endif
	
	// Build topological order of computational graph
    int topo_size;
    nnl2_ad_tensor** topo = nnl2_ad_build_topo(tensor, &topo_size);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(topo, "Failed to build topo in backpropagation");
	#endif

	// Initialize gradients to zero for all tensors that require gradients
    for (int i = 0; i < topo_size; i++) {
        if (topo[i]->requires_grad && !topo[i]->grad_initialized) {
            void* zero_value = nnl2_get_zero_value(topo[i]->grad->dtype);
            if (zero_value) {
                inplace_fill(topo[i]->grad, zero_value, topo[i]->grad->dtype);
                topo[i]->grad_initialized = true;
            }
        }
    }

	// Set initial gradient to 1 for the output tensor 
    void* grad_init_value = nnl2_get_one_value(tensor->grad->dtype);
    if (!grad_init_value) {
		NNL2_ERROR("Failed to initialize passed tensor with 1.0 in backpropagation");
        free(topo);
        return;
    }
    
    // Initialize the output tensor's gradient to 1 to start backpropagation
    bool success = inplace_fill(tensor->grad, grad_init_value, tensor->grad->dtype);
    if (!success) {
		NNL2_ERROR("Failed to initialize passed tensor with 1.0 in backpropagation");
        free(topo);
        return;
    }
	
	// Process nodes in reverse topological order
    for (int i = topo_size - 1; i >= 0; i--) {
        if (topo[i]->backward_fn) {
			// Execute backward function to compute gradients for this operation
            topo[i]->backward_fn(topo[i]); 			
        }
    }
	
	// Clear computational graph if not needed for future backward passes
	if(!retain_graph) nnl2_ad_clear_graph(topo, topo_size);
    
    // Free topological order array
    free(topo);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_BACKPROPAGATION_H **/	
