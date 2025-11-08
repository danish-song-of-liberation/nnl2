#ifndef NNL2_AD_LEAF_H
#define NNL2_AD_LEAF_H

/** @file nnl2_ad_leaf.h
 ** @date 2025
 ** @copyright MIT
 ** @brief Efficient and safe iterative leaf search for AD graphs
 **/

#define NNL2_AD_LEAF_STATIC_STACK_SIZE 256

/** @brief 
 * Finds the first leaf tensor in the computational graph using iterative DFS
 *
 ** @param tensor 
 * Pointer to the starting tensor in the graph
 *
 ** @return 
 * Pointer to the first found leaf tensor, or NULL if none is found
 */
NNL2_FORCE_INLINE static nnl2_ad_tensor* nnl2_ad_find_leaf(nnl2_ad_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    if(!tensor) {
		NNL2_ERROR("In function nnl2_ad_find_leaf (BPTT Core) passed tensor is NULL");
		return NULL;
	}

    // Static fast path stack 
    nnl2_ad_tensor* static_stack[NNL2_AD_LEAF_STATIC_STACK_SIZE];
    nnl2_ad_tensor** stack = static_stack;
    size_t stack_cap = NNL2_AD_LEAF_STATIC_STACK_SIZE;
    size_t top = 0;

    // Start DFS
    stack[top++] = tensor;

    while(top > 0) {
        nnl2_ad_tensor* current = stack[--top];
        if(!current) continue;

        if(current->is_leaf) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
			
            // Free heap memory if fallback occurred
            if(stack != static_stack) free(stack);
            return current;
        }

		if(!current->roots) continue;
        for(size_t i = 0; i < current->num_roots; ++i) {
            // Resize if stack is full
            if(top >= stack_cap) {
                size_t new_cap = stack_cap * 2;
                nnl2_ad_tensor** new_stack = (stack == static_stack)
                    ? malloc(new_cap * sizeof(*new_stack))
                    : realloc(stack, new_cap * sizeof(*new_stack));

				#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
					if(!new_stack) {
						NNL2_ERROR("Leaf search stack allocation failed at depth %zu", stack_cap);
						if(stack != static_stack) free(stack);
						return NULL;
					}
				#endif

                // Copy static data if switching from stack buffer
                if(stack == static_stack) {
                    memcpy(new_stack, static_stack, top * sizeof(*new_stack));
                }

                stack = new_stack;
                stack_cap = new_cap;
            }

            stack[top++] = current->roots[i];
        }
    }

    if(stack != static_stack) free(stack);

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return NULL;
}

#endif /** NNL2_AD_LEAF_H **/
