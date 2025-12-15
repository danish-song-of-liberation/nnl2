#ifndef NNL2_AD_VECTOR_AS_PARAMETER_H
#define NNL2_AD_VECTOR_AS_PARAMETER_H

/** @file nnl2_ad_vector_as_parameter.h
 ** @brief AD implementation for nnl2_vector_as_parameter
 ** @date 2025
 ** @copyright MIT
 **/

/** @brief 
 * Create a view AD tensor with given shape from a vector AD tensor starting at specified position
 * Does NOT track gradients (graph is not built) 
 *
 ** @param shape 
 * Array defining the new tensor shape
 *
 ** @param rank
 * Rank (number of dimensions) of the new tensor
 *
 ** @param start
 * Starting position in the vector (in elements)
 *
 ** @param vector
 * Source AD vector tensor to create view from
 *
 ** @return
 * Pointer to a new nnl2_ad_tensor representing a view of the original vector
 * NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if memory allocation fails or input is invalid
 *
 ** @note
 * This function creates a VIEW, not a copy. The returned tensor shares data with the original
 * Intended for GA 
 */
nnl2_ad_tensor* nnl2_ad_vector_as_parameter(int32_t* shape, int rank, size_t start, nnl2_ad_tensor* vector) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if(!vector) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, vector is NULL");
            return NULL;
        }
        
        if(!vector->data) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, vector->data is NULL");
            return NULL;
        }
        
        if(!shape) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, shape array is NULL");
            return NULL;
        }
        
        if(rank <= 0) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, rank must be positive");
            return NULL;
        }
        
        // Check that vector is actually a 1D tensor
        if(vector->data->rank != 1) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, input must be a vector (rank=1)");
            return NULL;
        }
        
        // Check start position is within bounds
        size_t vector_size = vector->data->shape[0];
        if(start >= vector_size) {
            NNL2_ERROR("In nnl2_ad_vector_as_parameter, start position exceeds vector size");
            return NULL;
        }
    #endif

    // Create view tensor using the base function
    nnl2_tensor* view_tensor = nnl2_vector_as_parameter(shape, rank, start, vector->data);
    
    if(!view_tensor) {
        NNL2_ERROR("In nnl2_ad_vector_as_parameter, nnl2_vector_as_parameter returned NULL");
        return NULL;
    }

    // Allocate AD tensor structure
    nnl2_ad_tensor* result = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
    if(!result) {
        NNL2_MALLOC_ERROR();
        nnl2_free_tensor(view_tensor);
        return NULL;
    }

    // Initialize AD tensor fields
    result->magic_number = TENSOR_MAGIC_ALIVE;
    result->data = view_tensor;
    result->grad = NULL;           // No gradient tracking for views
    result->requires_grad = false;
    result->grad_initialized = false;
    result->num_roots = 0;
    result->roots = NULL;
    result->backward_fn = NULL;
    result->is_leaf = true;        // Views are treated as leaves

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

#endif /** NNL2_AD_VECTOR_AS_PARAMETER_H **/
