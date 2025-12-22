#ifndef NNL2_RAND_INPLACE_H
#define NNL2_RAND_INPLACE_H

/** @file nnl2_rand_inplace.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains definition of functions that fills tensor with random values [0, 1] in-place
 **/
 
/** @brief
 * Fills the given tensor with random values from standard uniform distribution [0, 1] (in-place)
 *
 ** @details
 * The function fills an existing tensor with random numbers from a standard uniform distribution.
 * This is equivalent to nnl2_naive_uniform_inplace with from=0.0, to=1.0 but with a simpler API.
 *
 ** @param tensor 
 * Tensor to fill with random values
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If tensor is NULL
 *
 ** @exception NNL2Error 
 * If passed tensor with unknown type
 *
 ** @example
 * // Fill a tensor with random floats between 0.0 and 1.0
 * nnl2_tensor* tensor = nnl2_empty((int[]){2, 2}, 2, FLOAT32);
 * nnl2_naive_rand_inplace(tensor);
 *
 ** @see nnl2_rand
 ** @see nnl2_naive_uniform_inplace
 **/
void nnl2_naive_rand_inplace(nnl2_tensor* tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_naive_rand_inplace, passed tensor is NULL");
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    if(total_elems == 0) return; // If zero elems then early return
    
    // Filling with random values [0, 1]
    switch(tensor->dtype) {
        case FLOAT64: {
            nnl2_float64* data = (nnl2_float64*)tensor->data;
			
            for(size_t i = 0; i < total_elems; i++) 
                data[i] = (double)rand() / RAND_MAX;

            break;
        }
        
        case FLOAT32: {
            nnl2_float32* data = (nnl2_float32*)tensor->data;
			
            for(size_t i = 0; i < total_elems; i++) 
                data[i] = (float)rand() / RAND_MAX;

            break;
        }

        case INT32: {
            nnl2_int32* data = (nnl2_int32*)tensor->data;
			
            for(size_t i = 0; i < total_elems; i++) 
                data[i] = rand() % 2;
			
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(tensor->dtype);
            return;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for rand_inplace operation
 * @details
 * Array follows the common backend registration pattern for standard uniform 
 * random number generation operations (in-place version). Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for standard uniform distribution (in-place)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_rand_inplace
 */
Implementation rand_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_rand_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for rand_inplace operation
 * @ingroup backend_system
 */
randinplacefn rand_inplace;

/** 
 * @brief Makes the rand_inplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(rand_inplace);

/** 
 * @brief Sets the backend for rand_inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for rand_inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_rand_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(rand_inplace_backends, rand_inplace, backend_name, CURRENT_BACKEND(rand_inplace));
}

/** 
 * @brief Gets the name of the active backend for rand_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_rand_inplace_backend() {
    return CURRENT_BACKEND(rand_inplace);
}

/**
 * @brief Function declaration for getting all available rand_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(rand_inplace);

/**
 * @brief Function declaration for getting the number of available rand_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(rand_inplace);

#endif /** NNL2_RAND_INPLACE_H **/
