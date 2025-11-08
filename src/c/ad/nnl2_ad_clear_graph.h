#ifndef NNL2_AD_CLEAR_GRAPH_H
#define NNL2_AD_CLEAR_GRAPH_H

// NNL2

/** @file nnl2_ad_clear_graph.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Automatic differentiation computational graph clearing utilities
 **/
 
/** @brief 
 * Clears the computational graph for automatic differentiation
 *
 ** @param topo 
 * Array of pointers to tensors in topological order
 *
 ** @param topo_size 
 * Number of tensors in the topological array
 */
static NNL2_FORCE_INLINE void nnl2_ad_clear_graph(nnl2_ad_tensor** topo, int topo_size) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!topo || topo_size <= 0) return;
	#endif
	
	// Iterate through all tensors in topological order
    for(nnl2_ad_tensor** p = topo; p != topo + topo_size; ++p) {
        nnl2_ad_tensor* node = *p;
        
        node->backward_fn = NULL;

        if (node->roots) {
            free(node->roots);
            node->roots = NULL;
            node->num_roots = 0;
        }

        node->extra_multiplier = 1.0f;
        node->extra_bool = false;
        node->extra_correspondence = NULL;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_CLEAR_GRAPH_H **/
