#ifndef NNL2_AD_NN_GA_MUTATION_UNIFORM_H
#define NNL2_AD_NN_GA_MUTATION_UNIFORM_H

/** @file nnl2_ad_nn_ga_mutation_uniform.h
 ** @brief AD implementation for nnl2_nn_ga_mutation_uniform
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Performs uniform mutation operation on a parent AD tensor
 * Creates a child AD tensor where each element has a probability to be mutated
 * by adding a random value in range [-delta, delta] based on mutation rate
 * Does NOT track gradients (graph is not built)
 *
 ** @param tensor
 * Pointer to the parent AD tensor
 *
 ** @param mutate_rate
 * Probability (0.0 to 1.0) of mutating each element
 *
 ** @param delta
 * Maximum absolute value of mutation to be added to elements
 *
 ** @return
 * Pointer to a new nnl2_ad_tensor containing the result of uniform mutation
 * NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails or input is invalid
 *
 ** @note
 * This function creates a NEW tensor with mutated data
 * Gradient tracking is disabled for the result tensor
 * Intended for Genetic Algorithms where gradient computation is not required
 */
nnl2_ad_tensor* nnl2_ad_nn_ga_mutation_uniform(nnl2_ad_tensor* tensor, float mutate_rate, float delta) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if(!tensor) {
            NNL2_ERROR("In nnl2_ad_nn_ga_mutation_uniform, tensor cannot be NULL");
            return NULL;
        }
        
        if(!tensor->data) {
            NNL2_ERROR("In nnl2_ad_nn_ga_mutation_uniform, tensor data cannot be NULL");
            return NULL;
        }
        
        if(mutate_rate < 0.0f || mutate_rate > 1.0f) {
            NNL2_ERROR("In nnl2_ad_nn_ga_mutation_uniform, mutate_rate must be between 0.0 and 1.0");
            return NULL;
        }
        
        if(delta < 0.0f) {
            NNL2_ERROR("In nnl2_ad_nn_ga_mutation_uniform, delta must be non-negative");
            return NULL;
        }
    #endif

    // Perform mutation using the base function
    nnl2_tensor* mutated_tensor = nn_ga_mutation_uniform(tensor->data, mutate_rate, delta);
    
    if(!mutated_tensor) {
        NNL2_ERROR("In nnl2_ad_nn_ga_mutation_uniform, nn_ga_mutation_uniform returned NULL");
        return NULL;
    }

    // Allocate AD tensor structure
    nnl2_ad_tensor* result = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        nnl2_free_tensor(mutated_tensor);
        return NULL;
    }

    // Initialize AD tensor fields
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->data = mutated_tensor;
    result->grad = NULL;           // No gradient tracking for mutation results
    result->requires_grad = false; 
    result->grad_initialized = false;
    result->num_roots = 0;
    result->roots = NULL;
    result->backward_fn = NULL;    // No backward function needed
    result->is_leaf = true;        // Mutation results are treated as leaves

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

#endif /** NNL2_AD_NN_GA_MUTATION_UNIFORM_H **/
