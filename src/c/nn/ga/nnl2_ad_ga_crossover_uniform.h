#ifndef NNL2_AD_NN_GA_CROSSOVER_UNIFORM_H
#define NNL2_AD_NN_GA_CROSSOVER_UNIFORM_H

/** @file nnl2_ad_nn_ga_crossover_uniform.h
 ** @brief AD implementation for nnl2_nn_ga_crossover_uniform
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Performs uniform crossover operation between two parent AD tensors
 * Creates a child AD tensor where each element is randomly selected from 
 * either parent_x or parent_y based on the crossover rate
 * Does NOT track gradients (graph is not built)
 *
 ** @param parent_x
 * Pointer to the first parent AD tensor
 *
 ** @param parent_y
 * Pointer to the second parent AD tensor
 *
 ** @param crossover_rate
 * Probability (0.0 to 1.0) of selecting elements from parent_x
 *
 ** @return
 * Pointer to a new nnl2_ad_tensor containing the result of uniform crossover
 * NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails or input is invalid
 *
 ** @note
 * This function creates a NEW tensor with copied data
 * Gradient tracking is disabled for the result tensor
 * Intended for Genetic Algorithms where gradient computation is not required
 */
nnl2_ad_tensor* nnl2_ad_nn_ga_crossover_uniform(nnl2_ad_tensor* parent_x, nnl2_ad_tensor* parent_y, float crossover_rate) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if(!parent_x || !parent_y) {
            NNL2_ERROR("In nnl2_ad_nn_ga_naive_crossover_uniform, parent tensors cannot be NULL");
            return NULL;
        }
        
        if(!parent_x->data || !parent_y->data) {
            NNL2_ERROR("In nnl2_ad_nn_ga_naive_crossover_uniform, parent tensor data cannot be NULL");
            return NULL;
        }
        
        if(crossover_rate < 0.0f || crossover_rate > 1.0f) {
            NNL2_ERROR("In nnl2_ad_nn_ga_naive_crossover_uniform, crossover_rate must be between 0.0 and 1.0");
            return NULL;
        }
    #endif

    // Perform crossover using the base function
    nnl2_tensor* child_tensor = nn_ga_crossover_uniform(parent_x->data, parent_y->data, crossover_rate);
    
    if(!child_tensor) {
        NNL2_ERROR("In nnl2_ad_nn_ga_naive_crossover_uniform, nn_ga_crossover_uniform returned NULL");
        return NULL;
    }

    // Allocate AD tensor structure
    nnl2_ad_tensor* result = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        nnl2_free_tensor(child_tensor);
        return NULL;
    }

    // Initialize AD tensor fields
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->data = child_tensor;
    result->grad = NULL;           // No gradient tracking for crossover results
    result->requires_grad = false; 
    result->grad_initialized = false;
    result->num_roots = 0;
    result->roots = NULL;
    result->backward_fn = NULL;    // No backward function needed
    result->is_leaf = true;        // Crossover results are treated as leaves

    // Common initialization
    result->name = NULL;
    result->ts_type = nnl2_type_ad;
    result->visited_gen = 0;
    result->extra_multiplier = 1.0f;
    result->extra_bool = false;
    result->extra_correspondence = NULL;
    result->extra_field = NULL;
    result->extra_free = NULL;

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return result;
}

#endif /** NNL2_AD_NN_GA_CROSSOVER_UNIFORM_H **/
