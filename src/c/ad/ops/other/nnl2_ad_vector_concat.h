#ifndef NNL2_AD_VECTOR_CONCAT_H
#define NNL2_AD_VECTOR_CONCAT_H

/** @file nnl2_ad_concat.h
 ** @brief AD implementation for nnl2_vector_concat
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Concatenate an array of AD tensors into one flat AD tensor
 * Does NOT track gradients (graph is not built)
 *
 ** @param ad_tensors 
 * Array of nnl2_ad_tensor* to concatenate
 *
 ** @param count
 * Number of tensors in the array
 *
 ** @param dtype
 * Data type of the resulting tensor
 *
 ** @return
 * Pointer to a new nnl2_ad_tensor containing concatenated data
 * NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails or input is NULL
 *
 ** @note
 * This function is intended for GA / manual FNN usage,
 * gradients are NOT tracked
 */
nnl2_ad_tensor* nnl2_ad_vector_concat(nnl2_ad_tensor** ad_tensors, size_t count, nnl2_tensor_type dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety check
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if(!ad_tensors) {
            NNL2_ERROR("In nnl2_ad_naive_vector_concat, ad_tensors array is NULL");
            return NULL;
        }
    #endif

    nnl2_tensor** tensors = (nnl2_tensor**)malloc(count * sizeof(nnl2_tensor*));
    if(!tensors) {
        NNL2_MALLOC_ERROR();
        return NULL;
    }

    for (size_t i = 0; i < count; i++) {
        tensors[i] = ad_tensors[i]->data; 
    }

    nnl2_tensor* big_tensor = nnl2_vector_concat(tensors, count, dtype);
    free(tensors);

    if(!big_tensor) return NULL;

    nnl2_ad_tensor* result = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        nnl2_free_tensor(big_tensor);
        return NULL;
    }

    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->data = big_tensor;
    result->grad = NULL;           // no gradient
    result->requires_grad = false;
    result->grad_initialized = false;
    result->num_roots = 0;
    result->roots = NULL;
    result->backward_fn = NULL;
    result->is_leaf = true;

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

#endif /** NNL2_AD_VECTOR_CONCAT_H **/

