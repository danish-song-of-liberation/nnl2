#ifndef NNL2_CORRESPONDENCE_DIV_INPLACE_H
#define NNL2_CORRESPONDENCE_DIV_INPLACE_H

/** @brief 
 * Divides each element of a tensor by a scalar value (in-place).
 * 
 ** @param tensor 
 * Pointer to the tensor whose elements will be divided
 * 
 ** @param divisor 
 * Pointer to the scalar value to divide by
 */
void naive_div_divf_inplace(Tensor* tensor, void* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    switch(tensor->dtype) {
        case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double div = *((double*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float div = *((float*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
            break;
        }
        
        case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t div = *((int32_t*)divisor);
            for(size_t i = 0; i < total_elems; i++) cast_data[i] /= div;
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

#if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32

/** @brief
 * Threshold for enabling parallel execution of the
 * scalar division in-place operation
 */
#define NNL2_DIV_DIVF_INPLACE_PARALLEL_THRESHOLD 1000000

/** @brief 
 * Worker function for parallel double precision scalar division
 * 
 ** @param arg 
 * Pointer to divdivfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_divf_float64(void* arg);

/** @brief
 * Worker function for parallel single precision scalar division
 * 
 ** @param arg 
 * Pointer to divdivfinplace_ptask structure containing thread parameters  
 * 
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_divf_float32(void* arg);

/** @brief
 * Worker function for parallel integer scalar division
 * 
 ** @param arg 
 * Pointer to divdivfinplace_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pdiv_divf_int32(void* arg);

/** @brief
 * High-performance parallel implementation of in-place scalar division
 * 
 ** @param tensor 
 * Pointer to tensor that will be modified in-place
 *
 ** @param divisor 
 * Pointer to scalar value to divide each element by
 */
void nnl2_own_div_divf_inplace(Tensor* tensor, void* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "Tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor->data, "Tensor data is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(divisor, "Divisor pointer is NULL");
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    if(total_elems < NNL2_DIV_DIVF_INPLACE_PARALLEL_THRESHOLD) {
        naive_div_divf_inplace(tensor, divisor);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    TensorType dtype = tensor->dtype;
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own div scalar in-place, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    divdivfinplace_ptask tasks[num_threads];
    
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure divisor value based on data type
    switch(dtype) {
        case FLOAT64: {
            double div_val = *((double*)divisor);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned = is_aligned;
                tasks[i].divisor.float64_div = div_val;
            }
            break;
        }
		
        case FLOAT32: {
            float div_val = *((float*)divisor);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned = is_aligned;
                tasks[i].divisor.float32_div = div_val;
            }
            break;
        }
		
        case INT32: {
            int32_t div_val = *((int32_t*)divisor);
            for (size_t i = 0; i < num_threads; i++) {
                tasks[i].dtype = dtype;
                tasks[i].aligned = is_aligned;
                tasks[i].divisor.int32_div = div_val;
            }
            break;
        }
		
        default: {
            NNL2_TYPE_ERROR(dtype);
            return;
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].tensor_data = tensor->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        void* (*worker_func)(void*) = NULL;
        switch(dtype) {
            case FLOAT64: worker_func = nnl2_own_pdiv_divf_float64; break;
            case FLOAT32: worker_func = nnl2_own_pdiv_divf_float32; break;
            case INT32:   worker_func = nnl2_own_pdiv_divf_int32;   break;
			
            default: {
                NNL2_TYPE_ERROR(dtype);
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_div_divf_inplace");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_div_divf_inplace");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

void* nnl2_own_pdiv_divf_float64(void* arg) {
    divdivfinplace_ptask* task = (divdivfinplace_ptask*)arg;
    double* data = (double*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    double divisor = task->divisor.float64_div;
    
    __m256d v_divisor = _mm256_set1_pd(divisor);
    size_t i = start;
    
    if(task->aligned) {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            __m256d v_data = _mm256_load_pd(&data[i]);
            __m256d v_result = _mm256_div_pd(v_data, v_divisor);
            _mm256_store_pd(&data[i], v_result);
        }
    } else {
        for(; i + 3 < end; i += 4) {
            _mm_prefetch((char*)&data[i + 16], _MM_HINT_T0);
            __m256d v_data = _mm256_loadu_pd(&data[i]);
            __m256d v_result = _mm256_div_pd(v_data, v_divisor);
            _mm256_storeu_pd(&data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        data[i] /= divisor;
    }
    
    return NULL;
}

void* nnl2_own_pdiv_divf_float32(void* arg) {
    divdivfinplace_ptask* task = (divdivfinplace_ptask*)arg;
    float* data = (float*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    float divisor = task->divisor.float32_div;
    
    __m256 v_divisor = _mm256_set1_ps(divisor);
    size_t i = start;
    
    if(task->aligned) {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            __m256 v_data = _mm256_load_ps(&data[i]);
            __m256 v_result = _mm256_div_ps(v_data, v_divisor);
            _mm256_store_ps(&data[i], v_result);
        }
    } else {
        for(; i + 7 < end; i += 8) {
            _mm_prefetch((char*)&data[i + 32], _MM_HINT_T0);
            __m256 v_data = _mm256_loadu_ps(&data[i]);
            __m256 v_result = _mm256_div_ps(v_data, v_divisor);
            _mm256_storeu_ps(&data[i], v_result);
        }
    }
    
    for(; i < end; i++) {
        data[i] /= divisor;
    }
    
    return NULL;
}

void* nnl2_own_pdiv_divf_int32(void* arg) {
    divdivfinplace_ptask* task = (divdivfinplace_ptask*)arg;
    int32_t* data = (int32_t*)task->tensor_data;
    size_t start = task->start;
    size_t end = task->end;
    int32_t divisor = task->divisor.int32_div;
    
    size_t i = start;
    
    // Integer division uses scalar operations
    for(; i < end; i++) {
        data[i] /= divisor;
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place scalar division operation
 * @details
 * Array follows the common backend registration pattern for in-place scalar division
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for scalar division
 *  - nnl2_own: High-performance implementation with AVX256, pthread and prefetching
 * 
 * @see nnl2_naive
 * @see naive_div_divf_inplace
 * @see nnl2_own_div_divf_inplace
 */
Implementation div_divf_inplace_backends[] = {
    REGISTER_BACKEND(naive_div_divf_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #if defined(NNL2_PTHREAD_AVAILABLE) && defined(NNL2_AVX256_AVAILABLE) && TENSOR_MEM_ALIGNMENT == 32
        REGISTER_BACKEND(nnl2_own_div_divf_inplace, nnl2_own, NNL2_OWN_NAME),
    #endif
}; 

/**
 * @brief Function pointer for in-place scalar division operation
 * @ingroup backend_system 
 */
divdivfinplacefn div_divf_inplace;

/** 
 * @brief Sets the backend for in-place scalar division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for scalar division
 * @see SET_BACKEND_BY_NAME
 */
void set_div_divf_inplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(div_divf_inplace_backends, div_divf_inplace, backend_name);
}

#endif /** NNL2_CORRESPONDENCE_DIV_INPLACE_H **/
