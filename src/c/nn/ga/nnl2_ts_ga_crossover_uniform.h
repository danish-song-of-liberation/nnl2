#ifndef NNL2_NN_GA_CROSSOVER_UNIFORM_H
#define NNL2_NN_GA_CROSSOVER_UNIFORM_H

// NNL2

/** @brief
 * Performs uniform crossover operation between two parent tensors
 * 
 ** @param parent_x
 * Pointer to the first parent tensor
 *
 ** @param parent_y
 * Pointer to the second parent tensor
 *
 ** @param crossover_rate
 * Probability (0.0 to 1.0) of selecting elements from parent_x
 *
 ** @return nnl2_tensor*
 * Pointer to a new tensor containing the result of uniform crossover
 * (or NULL in case of failure)
 */
nnl2_tensor* nnl2_nn_ga_naive_crossover_uniform(nnl2_tensor* parent_x, nnl2_tensor* parent_y, float crossover_rate) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_tensor* child = nnl2_empty_like(parent_x);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(child == NULL) {
			NNL2_ERROR("In function nnl2_nn_ga_naive_crossover_uniform, failed to allocate child tensor. returning NULL");
			return NULL;
		}
	#endif 
	
	size_t numel = product(parent_x -> shape, parent_x -> rank);
	
	switch(child -> dtype) {
        case FLOAT64: {
            nnl2_float64* parent_x_data = (nnl2_float64*)parent_x -> data;
            nnl2_float64* parent_y_data = (nnl2_float64*)parent_y -> data;
            nnl2_float64* child_data = (nnl2_float64*)child -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < crossover_rate) {
					child_data[it] = parent_x_data[it];
				} else {
					child_data[it] = parent_y_data[it];
				}
			}
			
			break;
        }
		
		case FLOAT32: {
            nnl2_float32* parent_x_data = (nnl2_float32*)parent_x -> data;
            nnl2_float32* parent_y_data = (nnl2_float32*)parent_y -> data;
            nnl2_float32* child_data = (nnl2_float32*)child -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < crossover_rate) {
					child_data[it] = parent_x_data[it];
				} else {
					child_data[it] = parent_y_data[it];
				}
			}
			
			break;
        }
		
		case INT32: {
            nnl2_int32* parent_x_data = (nnl2_int32*)parent_x -> data;
            nnl2_int32* parent_y_data = (nnl2_int32*)parent_y -> data;
            nnl2_int32* child_data = (nnl2_int32*)child -> data;
			
			for(size_t it = 0; it < numel; it++) {
				if((float)rand() / RAND_MAX < crossover_rate) {
					child_data[it] = parent_x_data[it];
				} else {
					child_data[it] = parent_y_data[it];
				}
			}
			
			break;
        }
		
		default: {
            NNL2_TYPE_ERROR(child -> dtype);
            return NULL;
        }
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return child;
}

/**
 * @ingroup nn_ga_backend_system
 * @brief Backend implementations for uniform crossover operation
 * @details
 * Array follows the common backend registration pattern for genetic algorithm
 * crossover operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for uniform crossover
 * 
 * @see nnl2_naive
 * @see nnl2_nn_ga_naive_crossover_uniform
 */
Implementation nn_ga_crossover_uniform_backends[] = {
    REGISTER_BACKEND(nnl2_nn_ga_naive_crossover_uniform, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for uniform crossover operation
 * @ingroup nn_ga_backend_system
 */
nn_ga_crossover_uniform_fn nn_ga_crossover_uniform;

/** 
 * @brief Sets the backend for uniform crossover operation
 * @ingroup nn_ga_backend_system
 * @param backend_name Name of the backend to activate for uniform crossover
 * @see SET_BACKEND_BY_NAME
 * @see nn_ga_crossover_uniform_backends
 */
void set_nn_ga_crossover_uniform_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nn_ga_crossover_uniform_backends, nn_ga_crossover_uniform, backend_name);
}

#endif /** NNL2_NN_GA_CROSSOVER_UNIFORM_H **/
