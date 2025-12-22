#ifndef NNL2_RAND_H
#define NNL2_RAND_H

/** @brief
 * Creates a tensor with random numbers from uniform distribution [0, 1)
 *
 ** @details
 * The function generates random numbers from a standard uniform distribution.
 * This is equivalent to naive_uniform with from=0.0, to=1.0 but with a simpler API.
 *
 ** @param shape
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank
 * Number of dimensions (length of shape array)
 *
 ** @param dtype
 * Data type of the tensor elements (must be floating point)
 *
 ** @return
 * Pointer to the newly created Tensor
 *
 ** @example
 * // Create a 2x2 tensor of random floats between 0.0 and 1.0
 * nnl2_tensor* random_tensor = nnl2_rand((int[]){2, 2}, 2, FLOAT32);
 *
 ** @see naive_uniform
 ** @see nnl2_empty
 **/
Tensor* naive_rand(int* shape, int rank, TensorType dtype) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(shape, rank, dtype);
    
    size_t total_elems = nnl2_product(shape, rank);
    if(total_elems == 0) return result;
    
    switch(dtype) {
        case FLOAT64: {
            double* data = (double*)result->data;
            for(size_t i = 0; i < total_elems; i++) {
                data[i] = (double)rand() / RAND_MAX;
            }
            break;
        }
        
        case FLOAT32: {
            float* data = (float*)result->data;
            for(size_t i = 0; i < total_elems; i++) {
                data[i] = (float)rand() / RAND_MAX;
            }
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(dtype);
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
 * @brief Backend implementations for rand operation
 * @details
 * Array follows the common backend registration pattern for standard uniform 
 * random number generation. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for standard uniform distribution
 * 
 * @see nnl2_naive
 * @see naive_rand
 */
Implementation rand_backends[] = {
    REGISTER_BACKEND(naive_rand, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for rand operation
 * @ingroup backend_system 
 */
randfn nnl2_rand;

/** 
 * @brief Makes the rand backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(rand);

/** 
 * @brief Sets the backend for rand operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for rand
 * @see ESET_BACKEND_BY_NAME
 */
void set_rand_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(rand_backends, nnl2_rand, backend_name, current_backend(rand));
}

/** 
 * @brief Gets the name of the active backend for rand operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_rand_backend() {
    return current_backend(rand);
}

/** 
 * @brief Function declaration for getting all available rand backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(rand);

/**
 * @brief Function declaration for getting the number of available rand backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(rand);

#endif /** NNL2_RAND_H **/

