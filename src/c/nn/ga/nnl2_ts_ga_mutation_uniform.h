#ifndef NNL2_TS_GA_MUTATION_UNIFORM_H
#define NNL2_TS_GA_MUTATION_UNIFORM_H

// NNL2

/** @brief 
 * Performs uniform mutation on a tensor for genetic algorithm operations
 * 
 ** @param tensor 
 * Pointer to the input tensor to mutate
 *
 ** @param mutate_rate 
 * Probability of mutating each individual element (0.0 to 1.0)
 *
 ** @param delta 
 * Maximum absolute value of the random mutation to be added/subtracted
 */
nnl2_tensor* nnl2_nn_ga_naive_mutation_uniform(nnl2_tensor* tensor, float mutate_rate, float delta) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_tensor* result = nnl2_empty_like(tensor);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("In function nnl2_nn_ga_naive_mutation_uniform, failed to allocate result tensor. returning NULL");
			return NULL;
		}
	#endif 
	
	size_t numel = nnl2_product(tensor -> shape, tensor -> rank);
	
	switch(result -> dtype) {
        case FLOAT64: {
            nnl2_float64* tensor_data = (nnl2_float64*)tensor -> data;
            nnl2_float64* result_data = (nnl2_float64*)result -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < mutate_rate) {
					result_data[it] = (tensor_data[it] + ((((float)rand() / RAND_MAX) * 2 * delta) - delta));
				} else {
					result_data[it] = tensor_data[it];
				}
			}
			
			break;
        }
		
		case FLOAT32: {
            nnl2_float32* tensor_data = (nnl2_float32*)tensor -> data;
            nnl2_float32* result_data = (nnl2_float32*)result -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < mutate_rate) {
					result_data[it] = (tensor_data[it] + ((((float)rand() / RAND_MAX) * 2 * delta) - delta));
				} else {
					result_data[it] = tensor_data[it];
				}
			}
			
			break;
        }
		
		case INT32: {
            nnl2_int32* tensor_data = (nnl2_int32*)tensor -> data;
            nnl2_int32* result_data = (nnl2_int32*)result -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < mutate_rate) {
					result_data[it] = (tensor_data[it] + ((((float)rand() / RAND_MAX) * 2 * delta) - delta));
				} else {
					result_data[it] = tensor_data[it];
				}
			}
			
			break;
        }
		
		default: {
            NNL2_TYPE_ERROR(result -> dtype);
            return NULL;
        }
	}		
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return result;
}

/**
 * @ingroup nn_ga_backend_system
 * @brief Backend implementations for uniform mutation operation
 * @details
 * Array follows the common backend registration pattern for genetic algorithm
 * mutation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for uniform mutation
 * 
 * @see nnl2_nn_ga_naive_mutation_uniform
 */
Implementation nn_ga_mutation_uniform_backends[] = {
    REGISTER_BACKEND(nnl2_nn_ga_naive_mutation_uniform, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for uniform mutation operation
 * @ingroup nn_ga_backend_system
 */
nn_ga_mutation_uniform_fn nn_ga_mutation_uniform;

/** 
 * @brief Sets the backend for uniform mutation operation
 * @ingroup nn_ga_backend_system
 * @param backend_name Name of the backend to activate for uniform mutation
 * @see SET_BACKEND_BY_NAME
 * @see nn_ga_mutation_uniform_backends
 */
void set_nn_ga_mutation_uniform_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nn_ga_mutation_uniform_backends, nn_ga_mutation_uniform, backend_name);
}

#endif /** NNL2_TS_GA_MUTATION_UNIFORM_H **/
