#ifndef NNL2_XAVIER_INPLACE_H
#define NNL2_XAVIER_INPLACE_H

/** @brief
 * In-place Xavier initialization of a tensor
 *
 * Standard deviation is calculated as: gain * sqrt(distribution / (in + out))
 *
 ** @param tensor
 * Pointer to the tensor to initialize
 *
 ** @param in
 * Number of input neurons
 *
 ** @param out
 * Number of output neurons
 *
 ** @param gain
 * Gain factor
 *
 ** @param distribution
 * Distribution parameter (usually 2.0 or 6.0)
 *
 ** @note
 * Integer data types are not supported
 *
 ** @see RAND_MAX
 ** @see nnl2_product
 ** @see sqrt
 **/
void nnl2_naive_xavier_inplace(nnl2_tensor* tensor, int in, int out, float gain, float distribution) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In nnl2_naive_xavier_inplace, tensor is NULL");
    #endif

    if(tensor->dtype == INT32) {
        NNL2_FATAL("INT32 can't be used for Xavier initialization");
        return;
    }

    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;

    float scale_param = gain * sqrt(distribution / (in + out));
    
    if(fabsf(distribution - 6.0f) < 1e-6f) {
        double from = (double)-scale_param;
        double to = (double)scale_param;
        
        uniform_inplace(tensor, &from, &to);
        
    } else {
        randn_inplace(tensor, 0.0, (double)scale_param);
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place Xavier operation
 * @details
 * Array follows the common backend registration pattern for Xavier initialization
 * operations. Currently registered backends:
 *  - nnl2_naive_xavier_inplace: Basic reference implementation for in-place Xavier initialization
 *
 * @see nnl2_naive
 * @see nnl2_naive_xavier_inplace
 */
Implementation xavier_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_xavier_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place Xavier operation
 * @ingroup backend_system
 */
xavierinplacefn xavier_inplace;

/**
 * @brief Makes the in-place Xavier backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(xavier_inplace);

/**
 * @brief Sets the backend for in-place Xavier operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place Xavier
 * @see ESET_BACKEND_BY_NAME
 */
void set_xavier_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(xavier_inplace_backends, xavier_inplace, backend_name, CURRENT_BACKEND(xavier_inplace));
}

/**
 * @brief Gets the name of the active backend for in-place Xavier operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_xavier_inplace_backend() {
    return CURRENT_BACKEND(xavier_inplace);
}

/**
 * @brief Function declaration for getting all available in-place Xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(xavier_inplace);

/**
 * @brief Function declaration for getting the number of available in-place Xavier backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(xavier_inplace);

#endif /** NNL2_XAVIER_INPLACE_H **/
