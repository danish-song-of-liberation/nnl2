#ifndef NNL2_TANH_H
#define NNL2_TANH_H

#include <stdbool.h>

/** @brief
 * Calculates the hyperbolic tangent (tanh) for each tensor element
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param approx
 * Whether to use approximation for faster computation
 * true: uses fast approximation tanh(x) ~= x * (27 + x*x) / (27 + 9*x*x)
 * false: uses exact calculation tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
 *
 ** @return
 * Pointer to a new tensor containing the tanh-activated values
 * Returns NULL in case of failure
 *
 ** @details
 * The tanh function maps input values to the range (-1, 1).
 * For integer tensors (INT32), the output tensor will be promoted to FLOAT64.
 * Uses different approximation strategies based on the data type and approximation flag.
 *
 ** @see nnl2_product
 ** @see nnl2_empty
 ** @see tanh
 ** @see tanhf
 **/
Tensor* naive_tanh(Tensor* tensor, bool approx) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(tensor == NULL) {
            NNL2_ERROR("Passed tensor is NULL");
            return NULL;
        }
    #endif

	int total_elems = nnl2_product(tensor->shape, tensor->rank);
	TensorType dtype = tensor->dtype;
	
	if(dtype == INT32) dtype = FLOAT64;
	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
	if(total_elems == 0) return result;
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	switch(tensor->dtype) {
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;
			double* cast_data_r = (double*)data_r;
            if (approx) {
                for(int i = 0; i < total_elems; i++) {
                    double x = (double)cast_data_t[i];
                    double x2 = x * x;
                    // Rational approximation: tanh(x) ~= x * (27 + x^2) / (27 + 9 * x^2)
                    cast_data_r[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
                }
            } else {
                for(int i = 0; i < total_elems; i++) {
                    cast_data_r[i] = tanh((double)cast_data_t[i]);
                }
            }
			break;
		}
		
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
            if (approx) {
                for(int i = 0; i < total_elems; i++) {
                    double x = cast_data_t[i];
                    double x2 = x * x;
                    // Rational approximation: tanh(x) ~= x * (27 + x^2) / (27 + 9 * x^2)
                    cast_data_r[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
                }
            } else {
                for(int i = 0; i < total_elems; i++) {
                    cast_data_r[i] = tanh(cast_data_t[i]);
                }
            }
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
            if (approx) {
                for(int i = 0; i < total_elems; i++) {
                    float x = cast_data_t[i];
                    float x2 = x * x;
                    // Rational approximation: tanh(x) ~=  x * (27 + x^2) / (27 + 9 * x^2)
                    cast_data_r[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
                }
            } else {
                for(int i = 0; i < total_elems; i++) {
                    cast_data_r[i] = tanhf(cast_data_t[i]);
                }
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

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Parallel implementation of tanh activation function
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param approx
 * Whether to use approximation for faster computation
 *
 ** @return
 * Pointer to a new tensor containing the tanh-activated values
 * Returns NULL in case of failure
 *
 ** @details
 * Uses multi-threading for optimal performance on large tensors.
 * Pure parallel version without any SIMD optimizations.
 **/
Tensor* nnl2_own_tanh(Tensor* tensor, bool approx) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(tensor == NULL) {
            NNL2_ERROR("Passed tensor is NULL");
            return NULL;
        }
    #endif

    int total_elems = nnl2_product(tensor->shape, tensor->rank);
    TensorType dtype = tensor->dtype;
    
    if(dtype == INT32) dtype = FLOAT64;
    
    Tensor* result = nnl2_empty(tensor->shape, tensor->rank, dtype);
    if(total_elems == 0) return result;

    // For small tensors, use naive implementation
    if(total_elems < 10000) {
        Tensor* naive_result = naive_tanh(tensor, approx);
        if (naive_result) {
            size_t data_size = total_elems * get_dtype_size(dtype);
            memcpy(result->data, naive_result->data, data_size);
            nnl2_free_tensor(naive_result);
        }
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    // Worker function for parallel execution
    void* nnl2_tanh_worker(void* arg) {
        tanh_ptask* task = (tanh_ptask*)arg;
        size_t start = task->start_idx;
        size_t end = task->end_idx;
        
        switch(task->dtype) {
            case INT32: {
                int32_t* src_data = (int32_t*)task->src_data;
                double* dst_data = (double*)task->dst_data;
				
                if (task->approx) {
                    for(size_t i = start; i < end; i++) {
                        double x = (double)src_data[i];
                        double x2 = x * x;
                        dst_data[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
                    }
                } else {
                    for(size_t i = start; i < end; i++) {
                        dst_data[i] = tanh((double)src_data[i]);
                    }
                }
				
                break;
            }
            
            case FLOAT64: {
                double* src_data = (double*)task->src_data;
                double* dst_data = (double*)task->dst_data;
				
                if (task->approx) {
                    for(size_t i = start; i < end; i++) {
                        double x = src_data[i];
                        double x2 = x * x;
                        dst_data[i] = x * (27.0 + x2) / (27.0 + 9.0 * x2);
                    }
                } else {
                    for(size_t i = start; i < end; i++) {
                        dst_data[i] = tanh(src_data[i]);
                    }
                }
				
                break;
            }
            
            case FLOAT32: {
                float* src_data = (float*)task->src_data;
                float* dst_data = (float*)task->dst_data;
				
                if (task->approx) {
                    for(size_t i = start; i < end; i++) {
                        float x = src_data[i];
                        float x2 = x * x;
                        dst_data[i] = x * (27.0f + x2) / (27.0f + 9.0f * x2);
                    }
                } else {
                    for(size_t i = start; i < end; i++) {
                        dst_data[i] = tanhf(src_data[i]);
                    }
                }
				
                break;
            }
            
            default: break;
        }
        return NULL;
    }
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    tanh_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].approx = approx;
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        int status = pthread_create(&threads[i], NULL, nnl2_tanh_worker, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_tanh");
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

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for tanh operation
 * @details
 * Array follows the common backend registration pattern for hyperbolic tangent 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tanh activation function
 *  - nnl2_own: Parallel implementation for tanh activation
 * 
 * @see nnl2_naive
 * @see naive_tanh
 */
Implementation tanh_backends[] = {
	REGISTER_BACKEND(naive_tanh, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_tanh, nnl2_own, NNL2_OWN_NAME),
    #endif
};	

/**
 * @brief Function pointer for tanh operation
 * @ingroup backend_system 
 */
tanhfn nnl2_tanh;

/** 
 * @brief Makes the tanh backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(tanh);

/** 
 * @brief Sets the backend for tanh operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for tanh
 * @see ESET_BACKEND_BY_NAME
 */
void set_tanh_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(tanh_backends, nnl2_tanh, backend_name, CURRENT_BACKEND(tanh));
}

/** 
 * @brief Gets the name of the active backend for tanh operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_tanh_backend() {
	return CURRENT_BACKEND(tanh);
}

/** 
 * @brief Function declaration for getting all available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(tanh);

/**
 * @brief Function declaration for getting the number of available tanh backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(tanh);

#endif /** NNL2_TANH_H **/
