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
        
		case INT32: {
            int32_t* data = (int32_t*)result->data;
            for(size_t i = 0; i < total_elems; i++) {
                data[i] = rand() % 2; 
            }
            break;
        }
        
        case INT64: {
            int64_t* data = (int64_t*)result->data;
            for(size_t i = 0; i < total_elems; i++) {
                data[i] = rand() % 2; 
            }
            break;
        }
		
		case INT8: {
			int8_t* data = (int8_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case INT16: {
			int16_t* data = (int16_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case UINT8: {
			uint8_t* data = (uint8_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case UINT16: {
			uint16_t* data = (uint16_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case UINT32: {
			uint32_t* data = (uint32_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case UINT64: {
			uint64_t* data = (uint64_t*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = rand() % 2; 
			}
			break;
		}

		case BOOL: {
			nnl2_bool* data = (nnl2_bool*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = (rand() % 2) ? true : false;
			}
			break;
		}

		case INT128: {
			nnl2_int128* data = (nnl2_int128*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = (nnl2_int128)(rand() % 2);
			}
			break;
		}

		case UINT128: {
			nnl2_uint128* data = (nnl2_uint128*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = (nnl2_uint128)(rand() % 2);
			}
			break;
		}

		case FLOAT128: {
			nnl2_float128* data = (nnl2_float128*)result->data;
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = (nnl2_float128)((double)rand() / RAND_MAX);
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
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(rand);

/** 
 * @brief Sets the backend for rand operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for rand
 * @see ESET_BACKEND_BY_NAME
 */
void set_rand_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(rand_backends, nnl2_rand, backend_name, CURRENT_BACKEND(rand));
}

/** 
 * @brief Gets the name of the active backend for rand operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_rand_backend() {
    return CURRENT_BACKEND(rand);
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

