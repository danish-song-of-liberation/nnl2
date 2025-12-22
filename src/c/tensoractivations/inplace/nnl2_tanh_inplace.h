#ifndef NNL2_TANH_INPLACE_H
#define NNL2_TANH_INPLACE_H

/** @brief
 * Calculates the hyperbolic tangent for all tensor elements in place
 *
 ** @details
 * Uses either exact calculation or fast rational approximation
 *
 ** @param tensor
 * A pointer to a tensor for processing
 *
 ** @param approx
 * Whether to use approximation for faster computation
 * true: uses fast approximation tanh(x) ~= x * (27 + x*x) / (27 + 9*x*x)
 * false: uses exact calculation tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 *
 ** @note
 * Does not work with integer data types
 * 
 ** @see nnl2_product
 ** @see tanh
 ** @see tanhf
 **/
void naive_tanhinplace(Tensor* tensor, bool approx) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	int total_elems = nnl2_product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If tensor is empty then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;
			if (approx) {
				for(int i = 0; i < total_elems; i++) {
					double x = cast_data[i];
					double x2 = x * x;
					cast_data[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
				}
			} else {
				for(int i = 0; i < total_elems; i++) cast_data[i] = tanh(cast_data[i]);
			}
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;
			if (approx) {
				for(int i = 0; i < total_elems; i++) {
					float x = cast_data[i];
					float x2 = x * x;
					cast_data[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
				}
			} else {
				for(int i = 0; i < total_elems; i++) cast_data[i] = tanhf(cast_data[i]);
			}
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Tanh (in-place) cannot be applied to the provided tensor");
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

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Parallel implementation of tanh activation function in-place
 *
 ** @param tensor
 * A pointer to a tensor for processing
 *
 ** @param approx
 * Whether to use approximation for faster computation
 *
 ** @details
 * Uses multi-threading for optimal performance on large tensors.
 * Pure parallel version without any SIMD optimizations.
 **/
void nnl2_own_tanhinplace(Tensor* tensor, bool approx) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    int total_elems = nnl2_product(tensor->shape, tensor->rank);	
    if(total_elems == 0) return;

    // For small tensors, use naive implementation
    if(total_elems < 10000) {
        naive_tanhinplace(tensor, approx);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    // Worker function for parallel execution
    void* nnl2_tanhinplace_worker(void* arg) {
        tanhinplace_ptask* task = (tanhinplace_ptask*)arg;
        size_t start = task->start_idx;
        size_t end = task->end_idx;
        
        switch(task->dtype) {
            case FLOAT64: {
                double* data = (double*)task->data;
                if (task->approx) {
                    for(size_t i = start; i < end; i++) {
                        double x = data[i];
                        double x2 = x * x;
                        data[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
                    }
                } else {
                    for(size_t i = start; i < end; i++) {
                        data[i] = tanh(data[i]);
                    }
                }
                break;
            }
            
            case FLOAT32: {
                float* data = (float*)task->data;
                if (task->approx) {
                    for(size_t i = start; i < end; i++) {
                        float x = data[i];
                        float x2 = x * x;
                        data[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
                    }
                } else {
                    for(size_t i = start; i < end; i++) {
                        data[i] = tanhf(data[i]);
                    }
                }
                break;
            }
            
            default:
                break;
        }
        return NULL;
    }
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    tanhinplace_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].approx = approx;
        tasks[i].data = tensor->data;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        int status = pthread_create(&threads[i], NULL, nnl2_tanhinplace_worker, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_tanhinplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for tanh in-place operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * in-place operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 *  - nnl2_own: Parallel implementation for tanh activation
 * 
 * @see nnl2_naive
 * @see naive_tanhinplace
 */
Implementation tanhinplace_backends[] = {
	REGISTER_BACKEND(naive_tanhinplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_tanhinplace, nnl2_own, NNL2_OWN_NAME),
    #endif
};	

/**
 * @brief Function pointer for tanh in-place operation
 * @ingroup backend_system 
 */
tanhinplacefn tanhinplace;

/** 
 * @brief Makes the tanh in-place backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(tanhinplace);

/** 
 * @brief Sets the backend for tanh in-place operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh in-place
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanhinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanhinplace_backends, tanhinplace, backend_name, CURRENT_BACKEND(tanhinplace));
}

/** 
 * @brief Gets the name of the active backend for tanh in-place operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanhinplace_backend() {
	return CURRENT_BACKEND(tanhinplace);
}

/** 
 * @brief Function declaration for getting all available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanhinplace);

/**
 * @brief Function declaration for getting the number of available tanh in-place backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanhinplace);

#endif /** NNL2_TANH_INPLACE_H **/
