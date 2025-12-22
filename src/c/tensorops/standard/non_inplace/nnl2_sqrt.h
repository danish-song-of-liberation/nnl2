#ifndef NNL2_SQRT_H
#define NNL2_SQRT_H

/** @brief
 * Creates a new tensor with element-wise square root of the input tensor.
 * Each element is computed as: out[i] = sqrt(tensor[i])
 *
 ** @param tensor
 * Input tensor
 *
 ** @return
 * New tensor with square root values
 *
 ** @note
 * Works out-of-place (does not modify the input tensor)
 * For negative values behavior depends on dtype (NaN for floats, undefined for integers)
 *
 ** @see nnl2_tensor
 * @see nnl2_empty
 * @see nnl2_product
 */
nnl2_tensor* nnl2_naive_sqrt(const nnl2_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks at maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "nnl2_tensor is NULL (.sqrt)", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "nnl2_tensor's data is NULL (.sqrt)", NULL);
    #endif

    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);

    // Allocate new tensor of the same shape and dtype
    nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    if (total_elems == 0) return result;

    void* data_in  = tensor->data;
    void* data_out = result->data;

    switch (tensor->dtype) {
        case FLOAT64: {
            nnl2_float64* src = (nnl2_float64*)data_in;
            nnl2_float64* dst = (nnl2_float64*)data_out;
            for (size_t i = 0; i < total_elems; i++) dst[i] = sqrt(src[i]);
            break;
        }

        case FLOAT32: {
            nnl2_float32* src = (nnl2_float32*)data_in;
            nnl2_float32* dst = (nnl2_float32*)data_out;
            for (size_t i = 0; i < total_elems; i++) dst[i] = sqrtf(src[i]);
            break;
        }

        case INT32: {
            nnl2_int32* src = (nnl2_int32*)data_in;
            nnl2_int32* dst = (nnl2_int32*)data_out;
            for (size_t i = 0; i < total_elems; i++) {
                // For integers, we convert to float for sqrt then back to int
                dst[i] = (nnl2_int32)sqrt((double)src[i]);
            }
            break;
        }

        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            nnl2_free_tensor(result);
            return NULL;
        }
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for out-of-place sqrt operation
 * @details
 * Array follows the standard backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_sqrt: Basic reference implementation
 *
 * @see nnl2_naive_sqrt
 */
nnl2_runtime_implementation sqrt_backends[] = {
    REGISTER_BACKEND(nnl2_naive_sqrt, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for out-of-place sqrt operation
 * @ingroup backend_system
 */
sqrtfn nnl2_sqrt;

/**
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sqrt);

/**
 * @brief Sets the backend for out-of-place sqrt operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_sqrt_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sqrt_backends, nnl2_sqrt, backend_name, CURRENT_BACKEND(sqrt));
}

/**
 * @brief Gets the name of the active backend for out-of-place sqrt operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_sqrt_backend() {
    return CURRENT_BACKEND(sqrt);
}

/**
 * @brief Function declaration for getting all `sqrt` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sqrt);

/**
 * @brief Function declaration for getting the number of all `sqrt` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sqrt);

#endif /** NNL2_SQRT_H **/
