#ifndef NNL2_COS_H
#define NNL2_COS_H

#include <math.h>

/** @brief
 * Creates a new tensor with element-wise cosine of the input tensor.
 * Each element is computed as out[i] = cos(tensor[i])
 *
 ** @param tensor
 * Input tensor
 *
 ** @return
 * New tensor with cosine values
 *
 ** @note
 * Works out-of-place (does not modify the input tensor)
 *
 ** @see nnl2_tensor
 ** @see nnl2_empty
 ** @see nnl2_product
 */
nnl2_tensor* nnl2_naive_cos(const nnl2_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks at maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "nnl2_tensor is NULL (.cos)", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "nnl2_tensor's data is NULL (.cos)", NULL);
    #endif

    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);

    nnl2_tensor_type result_dtype = tensor->dtype;

    if (tensor->dtype == INT32) {
        result_dtype = FLOAT64;
    }

    // Allocate new tensor of the same shape, but possibly different dtype
    nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, result_dtype);
    if (!result) {
        NNL2_ERROR("Failed to allocate result tensor in nnl2_naive_cos");
        return NULL;
    }
    
    if (total_elems == 0) return result;

    void* data_in  = tensor->data;
    void* data_out = result->data;

    switch (tensor->dtype) {
        case FLOAT64: {
            nnl2_float64* src = (nnl2_float64*)data_in;
            nnl2_float64* dst = (nnl2_float64*)data_out;
            for (size_t i = 0; i < total_elems; i++) dst[i] = cos(src[i]);
            break;
        }

        case FLOAT32: {
            nnl2_float32* src = (nnl2_float32*)data_in;
            nnl2_float32* dst = (nnl2_float32*)data_out;
            for (size_t i = 0; i < total_elems; i++) dst[i] = cosf(src[i]);
            break;
        }
		
		case INT64: {
            nnl2_int64* src = (nnl2_int64*)data_in;
            nnl2_float64* dst = (nnl2_float64*)data_out;
            
            for (size_t i = 0; i < total_elems; i++) {
                double value = (double)src[i];
                dst[i] = cos(value);
            }
            
            break;
        }

        case INT32: {
            nnl2_int32* src = (nnl2_int32*)data_in;
            nnl2_float64* dst = (nnl2_float64*)data_out;
			
            for (size_t i = 0; i < total_elems; i++) {
                double value = (double)src[i];
                dst[i] = cos(value);
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
 * @brief Backend implementations for out-of-place cos operation
 * @details
 * Array follows the standard backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_cos: Basic reference implementation
 *
 * @see nnl2_naive_cos
 */
nnl2_runtime_implementation cos_backends[] = {
    REGISTER_BACKEND(nnl2_naive_cos, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for out-of-place cos operation
 * @ingroup backend_system
 */
cosfn nnl2_cos;

/**
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(cos);

/**
 * @brief Sets the backend for out-of-place cos operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_cos_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(cos_backends, nnl2_cos, backend_name, CURRENT_BACKEND(cos));
}

/**
 * @brief Gets the name of the active backend for out-of-place cos operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_cos_backend() {
    return CURRENT_BACKEND(cos);
}

/**
 * @brief Function declaration for getting all `cos` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(cos);

/**
 * @brief Function declaration for getting the number of all `cos` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(cos);

#endif /** NNL2_COS_H **/
