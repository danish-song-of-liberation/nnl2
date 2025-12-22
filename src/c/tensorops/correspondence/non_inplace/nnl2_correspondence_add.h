#ifndef NNL2_CORRESPONDENCE_ADD_H
#define NNL2_CORRESPONDENCE_ADD_H

/** @brief
 * Performs element-wise addition of a scalar increment to a tensor
 *
 ** @param tensor
 * Pointer to the input tensor to which the increment will be added
 *
 ** @param inc
 * Pointer to the scalar increment value
 *
 ** @return
 * Pointer to a new tensor containing the result of the addition operation 
 * (or NULL in case of fail)
 */
nnl2_tensor* naive_add_incf(nnl2_tensor* tensor, void* inc) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to allocate new tensor");
		}
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return result;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_original = (double*)tensor->data; 
			double* cast_data_result = (double*)result->data;    // Casting
			double increment = *((double*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment; // Assigment
			break;
		}
		
		case FLOAT32: {
			float* cast_data_original = (float*)tensor->data;
			float* cast_data_result = (float*)result->data;
			float increment = *((float*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
			break;
		}
		
		case INT32: {
			int32_t* cast_data_original = (int32_t*)tensor->data;
			int32_t* cast_data_result = (int32_t*)result->data;
			int32_t increment = *((int32_t*)inc);
			for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] + increment;
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

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * scalar addition with increment operation
 */
#define NNL2_ADD_INCF_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision scalar addition with increment
 * 
 ** @param arg 
 * Pointer to addincf_non_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_incf_float64_non_inplace(void* arg);

/** @brief
 * Worker function for parallel single precision scalar addition with increment
 * 
 ** @param arg 
 * Pointer to addincf_non_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_incf_float32_non_inplace(void* arg);

/** @brief
 * Worker function for parallel integer scalar addition with increment
 * 
 ** @param arg 
 * Pointer to addincf_non_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_padd_incf_int32_non_inplace(void* arg);

/** @brief
 * High-performance parallel implementation of scalar addition with increment
 * 
 ** @param tensor 
 * Pointer to input tensor
 *
 ** @param inc 
 * Pointer to scalar increment value
 *
 ** @return 
 * Pointer to a new tensor containing the result of the addition operation
 */
nnl2_tensor* nnl2_own_add_incf(nnl2_tensor* tensor, void* inc) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "nnl2_tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "nnl2_tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(inc, "Increment pointer is NULL", NULL);
    #endif
    
    nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    if(result == NULL) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("Failed to allocate new tensor");
        #endif
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return result;
    }
    
    if(total_elems < NNL2_ADD_INCF_PARALLEL_THRESHOLD) {
        nnl2_tensor* naive_result = naive_add_incf(tensor, inc);
        if(naive_result == NULL) {
            nnl2_free_tensor(result);
            return NULL;
        }
        nnl2_free_tensor(result);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return naive_result;
    }
    
    nnl2_tensor_type dtype = tensor->dtype;
    bool is_aligned_tensor = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned_result = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned_tensor) {
            NNL2_WARN("In nnl2_own add scalar, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
		
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own add scalar, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    addincf_non_inplace_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure increment value based on data type
    switch(dtype) {
        case FLOAT64: {
            double inc_val = *((double*)inc);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].increment.float64_inc = inc_val;
            }
			
            break;
        }
        case FLOAT32: {
            float inc_val = *((float*)inc);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].increment.float32_inc = inc_val;
            }
			
            break;
        }
        case INT32: {
            int32_t inc_val = *((int32_t*)inc);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].increment.int32_inc = inc_val;
            }
			
            break;
        }
        default: {
            NNL2_TYPE_ERROR(dtype);
            nnl2_free_tensor(result);
            return NULL;
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].tensor_data = tensor->data;
        tasks[i].result_data = result->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_padd_incf_float64_non_inplace; break;
            case FLOAT32: worker_func = nnl2_own_padd_incf_float32_non_inplace; break;
            case INT32:   worker_func = nnl2_own_padd_incf_int32_non_inplace;   break;
			
            default: {
                NNL2_TYPE_ERROR(dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_add_incf");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_add_incf");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_padd_incf_float64_non_inplace(void* arg) {
    addincf_non_inplace_ptask* task = (addincf_non_inplace_ptask*)arg;
    double* tensor_data = (double*)task->tensor_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    double increment = task->increment.float64_inc;
    
    __m256d v_increment = _mm256_set1_pd(increment);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_load_pd(&tensor_data[i]);
            __m256d v_result = _mm256_add_pd(v_tensor, v_increment);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_load_pd(&tensor_data[i]);
            __m256d v_result = _mm256_add_pd(v_tensor, v_increment);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_loadu_pd(&tensor_data[i]);
            __m256d v_result = _mm256_add_pd(v_tensor, v_increment);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_loadu_pd(&tensor_data[i]);
            __m256d v_result = _mm256_add_pd(v_tensor, v_increment);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] + increment;
    }
    
    return NULL;
}

void* nnl2_own_padd_incf_float32_non_inplace(void* arg) {
    addincf_non_inplace_ptask* task = (addincf_non_inplace_ptask*)arg;
    float* tensor_data = (float*)task->tensor_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    float increment = task->increment.float32_inc;
    
    __m256 v_increment = _mm256_set1_ps(increment);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_load_ps(&tensor_data[i]);
            __m256 v_result = _mm256_add_ps(v_tensor, v_increment);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_load_ps(&tensor_data[i]);
            __m256 v_result = _mm256_add_ps(v_tensor, v_increment);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_loadu_ps(&tensor_data[i]);
            __m256 v_result = _mm256_add_ps(v_tensor, v_increment);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_loadu_ps(&tensor_data[i]);
            __m256 v_result = _mm256_add_ps(v_tensor, v_increment);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] + increment;
    }
    
    return NULL;
}

void* nnl2_own_padd_incf_int32_non_inplace(void* arg) {
    addincf_non_inplace_ptask* task = (addincf_non_inplace_ptask*)arg;
    int32_t* tensor_data = (int32_t*)task->tensor_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t increment = task->increment.int32_inc;
    
    __m256i v_increment = _mm256_set1_epi32(increment);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_load_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_add_epi32(v_tensor, v_increment);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_load_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_add_epi32(v_tensor, v_increment);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_loadu_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_add_epi32(v_tensor, v_increment);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_loadu_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_add_epi32(v_tensor, v_increment);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] + increment;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for scalar addition with increment operation
 * @details
 * Array follows the common backend registration pattern for scalar addition with
 * increment operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar addition with increment
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_add_incf
 * @see nnl2_own_add_incf
 */
nnl2_runtime_implementation add_incf_backends[] = {
    REGISTER_BACKEND(naive_add_incf, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_add_incf, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for scalar addition with increment operation
 * @ingroup backend_system
 */
addincffn add_incf;

/** 
 * @brief Sets the backend for scalar addition with increment operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar addition with increment
 * @see SET_BACKEND_BY_NAME
 * @see add_incf_backends
 */
void set_add_incf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(add_incf_backends, add_incf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_ADD_H **/
