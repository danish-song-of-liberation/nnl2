#ifndef NNL2_CORRESPONDENCE_SUB_H
#define NNL2_CORRESPONDENCE_SUB_H

/** @brief
 * Performs element-wise subtraction of a scalar decrement from a tensor
 *
 ** @param tensor
 * Pointer to the input tensor from which the decrement will be subtracted
 *
 ** @param dec
 * Pointer to the scalar decrement value
 *
 ** @return
 * Pointer to a new tensor containing the result of the subtraction operation 
 * (or NULL in case of failure)
 */
nnl2_tensor* naive_sub_decf(const nnl2_tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(result == NULL) {
            NNL2_ERROR("Failed to allocate new tensor");
            return NULL;
        }
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return result;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data_original = (double*)tensor->data;
            double* cast_data_result = (double*)result->data;	// Casting
            double decrement = *((double*)dec); 
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement; // Assigment
            break;
        }
        
        case FLOAT32: {
            float* cast_data_original = (float*)tensor->data;
            float* cast_data_result = (float*)result->data;
            float decrement = *((float*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
            break;
        }
        
        case INT32: {
            int32_t* cast_data_original = (int32_t*)tensor->data;
            int32_t* cast_data_result = (int32_t*)result->data;
            int32_t decrement = *((int32_t*)dec);
            for(size_t i = 0; i < total_elems; i++) cast_data_result[i] = cast_data_original[i] - decrement;
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
 * scalar subtraction with decrement operation
 */
#define NNL2_SUB_DECF_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision scalar subtraction with decrement
 * 
 ** @param arg 
 * Pointer to subdecf_non_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_decf_float64_non_inplace(void* arg);

/** @brief
 * Worker function for parallel single precision scalar subtraction with decrement
 * 
 ** @param arg 
 * Pointer to subdecf_non_inplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_decf_float32_non_inplace(void* arg);

/** @brief
 * Worker function for parallel integer scalar subtraction with decrement
 * 
 ** @param arg 
 * Pointer to subdecf_non_inplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_psub_decf_int32_non_inplace(void* arg);

/** @brief
 * High-performance parallel implementation of scalar subtraction with decrement
 * 
 ** @param tensor 
 * Pointer to input tensor
 *
 ** @param dec 
 * Pointer to scalar decrement value
 *
 ** @return 
 * Pointer to a new tensor containing the result of the subtraction operation
 */
nnl2_tensor* nnl2_own_sub_decf(const nnl2_tensor* tensor, void* dec) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor, "nnl2_tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensor->data, "nnl2_tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(dec, "Decrement pointer is NULL", NULL);
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
    
    if(total_elems < NNL2_SUB_DECF_PARALLEL_THRESHOLD) {
        nnl2_tensor* naive_result = naive_sub_decf(tensor, dec);
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
            NNL2_WARN("In nnl2_own sub scalar, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
		
        if(!is_aligned_result) {
            NNL2_WARN("In nnl2_own sub scalar, result memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    subdecf_non_inplace_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure decrement value based on data type
    switch(dtype) {
        case FLOAT64: {
            double dec_val = *((double*)dec);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].decrement.float64_dec = dec_val;
            }
            break;
        }
        case FLOAT32: {
            float dec_val = *((float*)dec);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].decrement.float32_dec = dec_val;
            }
            break;
        }
        case INT32: {
            int32_t dec_val = *((int32_t*)dec);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned_tensor = is_aligned_tensor;
                tasks[i].aligned_result = is_aligned_result;
                tasks[i].decrement.int32_dec = dec_val;
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
            case FLOAT64: worker_func = nnl2_own_psub_decf_float64_non_inplace; break;
            case FLOAT32: worker_func = nnl2_own_psub_decf_float32_non_inplace; break;
            case INT32:   worker_func = nnl2_own_psub_decf_int32_non_inplace;   break;
			
            default: {
                NNL2_TYPE_ERROR(dtype);
                nnl2_free_tensor(result);
                return NULL;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sub_decf");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sub_decf");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_psub_decf_float64_non_inplace(void* arg) {
    subdecf_non_inplace_ptask* task = (subdecf_non_inplace_ptask*)arg;
    double* tensor_data = (double*)task->tensor_data;
    double* result_data = (double*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    double decrement = task->decrement.float64_dec;
    
    __m256d v_decrement = _mm256_set1_pd(decrement);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_load_pd(&tensor_data[i]);
            __m256d v_result = _mm256_sub_pd(v_tensor, v_decrement);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_load_pd(&tensor_data[i]);
            __m256d v_result = _mm256_sub_pd(v_tensor, v_decrement);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_loadu_pd(&tensor_data[i]);
            __m256d v_result = _mm256_sub_pd(v_tensor, v_decrement);
            _mm256_store_pd(&result_data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&tensor_data[i + 16], _MM_HINT_T0);
            __m256d v_tensor = _mm256_loadu_pd(&tensor_data[i]);
            __m256d v_result = _mm256_sub_pd(v_tensor, v_decrement);
            _mm256_storeu_pd(&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] - decrement;
    }
    
    return NULL;
}

void* nnl2_own_psub_decf_float32_non_inplace(void* arg) {
    subdecf_non_inplace_ptask* task = (subdecf_non_inplace_ptask*)arg;
    float* tensor_data = (float*)task->tensor_data;
    float* result_data = (float*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    float decrement = task->decrement.float32_dec;
    
    __m256 v_decrement = _mm256_set1_ps(decrement);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_load_ps(&tensor_data[i]);
            __m256 v_result = _mm256_sub_ps(v_tensor, v_decrement);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_load_ps(&tensor_data[i]);
            __m256 v_result = _mm256_sub_ps(v_tensor, v_decrement);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_loadu_ps(&tensor_data[i]);
            __m256 v_result = _mm256_sub_ps(v_tensor, v_decrement);
            _mm256_store_ps(&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256 v_tensor = _mm256_loadu_ps(&tensor_data[i]);
            __m256 v_result = _mm256_sub_ps(v_tensor, v_decrement);
            _mm256_storeu_ps(&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] - decrement;
    }
    
    return NULL;
}

void* nnl2_own_psub_decf_int32_non_inplace(void* arg) {
    subdecf_non_inplace_ptask* task = (subdecf_non_inplace_ptask*)arg;
    int32_t* tensor_data = (int32_t*)task->tensor_data;
    int32_t* result_data = (int32_t*)task->result_data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t decrement = task->decrement.int32_dec;
    
    __m256i v_decrement = _mm256_set1_epi32(decrement);
    size_t i = start;
    
    if(task->aligned_tensor && task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_load_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_sub_epi32(v_tensor, v_decrement);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else if(task->aligned_tensor) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_load_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_sub_epi32(v_tensor, v_decrement);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    } else if(task->aligned_result) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_loadu_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_sub_epi32(v_tensor, v_decrement);
            _mm256_store_si256((__m256i*)&result_data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&tensor_data[i + 32], _MM_HINT_T0);
            __m256i v_tensor = _mm256_loadu_si256((__m256i*)&tensor_data[i]);
            __m256i v_result = _mm256_sub_epi32(v_tensor, v_decrement);
            _mm256_storeu_si256((__m256i*)&result_data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        result_data[i] = tensor_data[i] - decrement;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for scalar subtraction with decrement operation
 * @details
 * Array follows the common backend registration pattern for scalar subtraction with
 * decrement operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar subtraction with decrement
 * 
 * @see nnl2_naive
 * @see naive_sub_decf
 */
nnl2_runtime_implementation sub_decf_backends[] = {
    REGISTER_BACKEND(naive_sub_decf, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_sub_decf, nnl2_own, NNL2_OWN_NAME),
    #endif
};    

/**
 * @brief Function pointer for scalar subtraction with decrement operation
 * @ingroup backend_system
 */
subdecffn sub_decf;

/** 
 * @brief Sets the backend for scalar subtraction with decrement operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar subtraction with decrement
 * @see SET_BACKEND_BY_NAME
 * @see sub_decf_backends
 */
void set_sub_decf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(sub_decf_backends, sub_decf, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_SUB_H **/
