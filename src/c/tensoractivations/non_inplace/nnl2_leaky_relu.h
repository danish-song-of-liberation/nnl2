#ifndef NNL2_LEAKY_RELY_H
#define NNL2_LEAKY_RELY_H

/** @brief
 * A function that does what you already know with a tensor, but documentation is required
 *
 ** @details
 * Applies Leaky ReLU (max(x * alpha, x)) function element-wise
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *
 ** @param tensor
 * Input tensor to apply Leaky ReLU to
 *
 ** @param alpha
 * Negative slope coefficient for values less than zero
 *
 ** @param save_type
 * Tries to preserve the initial tensor type if possible. It is recommended to set true
 *
 ** @return
 * New tensor with Leaky ReLU applied
 */
Tensor* naive_leakyrelu(Tensor* tensor, float alpha, bool save_type) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype); 
	
	if(total_elems == 0) return result; // If tensor is empty then return new empty tensor
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	bool float64_conversion = false;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float64(cast_data_t[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float32(cast_data_t[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			
			// Check if conversion to float64 is needed for int32
			for(int i = 0; i < total_elems; i++) {
				if(cast_data_t[i] < 0) {
					float result_val = cast_data_t[i] * alpha;
					
					// Check if result has fractional part
					if(fmodf(result_val, 1.0f) != 0.0f) {
						float64_conversion = true;
						break;
					}
				}
			}
			
			// Handle int32 with potential type conversion
			if(float64_conversion || !save_type) {
				nnl2_free_tensor(result); 
				
				result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
				data_r = result->data;
				
				double* cast_data_r_f64 = (double*)data_r;
				
				for(int i = 0; i < total_elems; i++) {
					if(cast_data_t[i] >= 0) {
						cast_data_r_f64[i] = (double)cast_data_t[i];
					} else {
						cast_data_r_f64[i] = (double)(cast_data_t[i] * alpha);
					}
				}
			} else {
				for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_int32(cast_data_t[i], alpha);
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return NULL;	
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}



#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Worker function for parallel double precision Leaky ReLU
 */
void* nnl2_simple_pleakyrelu_float64(void* arg) {
    leakyrelu_ptask* task = (leakyrelu_ptask*)arg;
    double* src_data = (double*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    double alpha = (double)task->alpha;
    
    for(size_t i = task->start_idx; i < task->end_idx; i++) {
        dst_data[i] = src_data[i] >= 0.0 ? src_data[i] : src_data[i] * alpha;
    }
    
    return NULL;
}

/** @brief
 * Worker function for parallel single precision Leaky ReLU
 */
void* nnl2_simple_pleakyrelu_float32(void* arg) {
    leakyrelu_ptask* task = (leakyrelu_ptask*)arg;
    float* src_data = (float*)task->src_data;
    float* dst_data = (float*)task->dst_data;
    float alpha = task->alpha;
    
    for(size_t i = task->start_idx; i < task->end_idx; i++) {
        dst_data[i] = src_data[i] >= 0.0f ? src_data[i] : src_data[i] * alpha;
    }
    
    return NULL;
}

/** @brief
 * Worker function for parallel integer Leaky ReLU with proper type checking
 */
void* nnl2_simple_pleakyrelu_int32(void* arg) {
    leakyrelu_ptask* task = (leakyrelu_ptask*)arg;
    int32_t* src_data = (int32_t*)task->src_data;
    int32_t* dst_data = (int32_t*)task->dst_data;
    float alpha = task->alpha;
    
    for(size_t i = task->start_idx; i < task->end_idx; i++) {
        if(src_data[i] >= 0) {
            dst_data[i] = src_data[i];
        } else {
            float result_val = src_data[i] * alpha;
            // Check if the result can be represented as int32 without fractional part
            if(fmodf(fabsf(result_val), 1.0f) == 0.0f && result_val >= INT32_MIN && result_val <= INT32_MAX) {
                dst_data[i] = (int32_t)result_val;
            } else {
                // If cannot be represented as int32, use 0 as fallback
                dst_data[i] = 0;
            }
        }
    }
    
    return NULL;
}

/** @brief
 * Worker function for parallel integer Leaky ReLU with conversion to float64
 */
void* nnl2_simple_pleakyrelu_int32_to_float64(void* arg) {
    leakyrelu_ptask* task = (leakyrelu_ptask*)arg;
    int32_t* src_data = (int32_t*)task->src_data;
    double* dst_data = (double*)task->dst_data;
    float alpha = task->alpha;
    
    for(size_t i = task->start_idx; i < task->end_idx; i++) {
        if(src_data[i] >= 0) {
            dst_data[i] = (double)src_data[i];
        } else {
            dst_data[i] = (double)(src_data[i] * alpha);
        }
    }
    
    return NULL;
}

/** @brief
 * Check if int32 tensor requires conversion to float64 for LeakyReLU
 */
static bool nnl2_leakyrelu_int32_needs_conversion(Tensor* tensor, float alpha) {
    int32_t* data = (int32_t*)tensor->data;
    size_t total_elems = product(tensor->shape, tensor->rank);
    
    for(size_t i = 0; i < total_elems; i++) {
        if(data[i] < 0) {
            float result_val = data[i] * alpha;
            if(fmodf(fabsf(result_val), 1.0f) != 0.0f) {
                return true;
            }
        }
    }
    return false;
}

/** @brief
 * Simple parallel implementation of Leaky ReLU activation function using raw pthread
 * 
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param alpha
 * Negative slope coefficient for values less than zero
 *
 ** @param save_type
 * Tries to preserve the initial tensor type if possible
 *
 ** @return
 * Pointer to a new tensor containing the Leaky ReLU-activated values
 */
Tensor* nnl2_simple_leakyrelu(Tensor* tensor, float alpha, bool save_type) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate total number of elements
    size_t total_elems = product(tensor->shape, tensor->rank);	
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    
    if(total_elems == 0) return result;
    
    // Handle int32 type conversion logic first
    if(tensor->dtype == INT32) {
        bool float64_conversion = nnl2_leakyrelu_int32_needs_conversion(tensor, alpha);
        
        if(float64_conversion || !save_type) {
            nnl2_free_tensor(result);
            result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64);
            
            // Use parallel processing for int32 to float64 conversion
            size_t num_threads = NNL2_NUM_THREADS;
            pthread_t threads[num_threads];
            leakyrelu_ptask tasks[num_threads];
            
            size_t chunk = total_elems / num_threads;
            size_t remainder = total_elems % num_threads;
            
            size_t current_start = 0;
            for (size_t i = 0; i < num_threads; i++) {
                size_t current_chunk = chunk + (i < remainder ? 1 : 0);
                
                tasks[i].dtype = FLOAT64;
                tasks[i].src_data = tensor->data;
                tasks[i].dst_data = result->data;
                tasks[i].alpha = alpha;
                tasks[i].inplace = false;
                tasks[i].start_idx = current_start;
                tasks[i].end_idx = current_start + current_chunk;
                
                int status = pthread_create(&threads[i], NULL, nnl2_simple_pleakyrelu_int32_to_float64, &tasks[i]);
                if(status != 0) {
                    NNL2_THREAD_CREATE_ERROR(status, "nnl2_simple_leakyrelu");
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
            return result;
        } else {
            // Use parallel processing for int32 without conversion
            size_t num_threads = NNL2_NUM_THREADS;
            pthread_t threads[num_threads];
            leakyrelu_ptask tasks[num_threads];
            
            size_t chunk = total_elems / num_threads;
            size_t remainder = total_elems % num_threads;
            
            size_t current_start = 0;
            for (size_t i = 0; i < num_threads; i++) {
                size_t current_chunk = chunk + (i < remainder ? 1 : 0);
                
                tasks[i].dtype = INT32;
                tasks[i].src_data = tensor->data;
                tasks[i].dst_data = result->data;
                tasks[i].alpha = alpha;
                tasks[i].inplace = false;
                tasks[i].start_idx = current_start;
                tasks[i].end_idx = current_start + current_chunk;
                
                int status = pthread_create(&threads[i], NULL, nnl2_simple_pleakyrelu_int32, &tasks[i]);
                if(status != 0) {
                    NNL2_THREAD_CREATE_ERROR(status, "nnl2_simple_leakyrelu");
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
            return result;
        }
    }
    
    // Parallel processing for float32, float64
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    leakyrelu_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].dtype = tensor->dtype;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
        tasks[i].alpha = alpha;
        tasks[i].inplace = false;
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(tensor->dtype) {
            case FLOAT64: worker_func = nnl2_simple_pleakyrelu_float64; break;
            case FLOAT32: worker_func = nnl2_simple_pleakyrelu_float32; break;
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_simple_leakyrelu");
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
    
    return result;
}

/** @brief
 * Simple in-place Leaky ReLU activation function using raw pthread
 * 
 ** @param tensor
 * Pointer to the input tensor (modified in-place)
 *
 ** @param alpha
 * Negative slope coefficient for values less than zero
 */
void nnl2_simple_leakyrelu_inplace(Tensor* tensor, float alpha) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);	
    if(total_elems == 0) return;
    
    // In-place doesn't support int32->float64 conversion
    if(tensor->dtype == INT32) {
        // Check if conversion would be needed
        if(nnl2_leakyrelu_int32_needs_conversion(tensor, alpha)) {
            NNL2_ERROR("In-place LeakyReLU cannot convert int32 to float64. Use out-of-place version.");
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            return;
        }
    }
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    leakyrelu_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].dtype = tensor->dtype;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = tensor->data; // Same pointer for in-place
        tasks[i].alpha = alpha;
        tasks[i].inplace = true;
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(tensor->dtype) {
            case FLOAT64: worker_func = nnl2_simple_pleakyrelu_float64; break;
            case FLOAT32: worker_func = nnl2_simple_pleakyrelu_float32; break;
            case INT32:   worker_func = nnl2_simple_pleakyrelu_int32;   break;
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_simple_leakyrelu_inplace");
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
 * @brief Backend implementations for LeakyReLU operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyrelu
 */
Implementation leakyrelu_backends[] = {
	REGISTER_BACKEND(naive_leakyrelu, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
	    REGISTER_BACKEND(nnl2_simple_leakyrelu, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for LeakyReLU operation
 * @ingroup backend_system 
 */
leakyrelufn leakyrelu;

/** 
 * @brief Makes the leakyrelu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(leakyrelu);

/** 
 * @brief Sets the backend for LeakyReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyrelu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyrelu_backends, leakyrelu, backend_name, current_backend(leakyrelu));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyrelu_backend() {
	return current_backend(leakyrelu);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyrelu);

/**
 * @brief Function declaration for getting the number of available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyrelu);

#endif /** NNL2_LEAKY_RELY_H **/
