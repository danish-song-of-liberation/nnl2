#ifndef NNL2_DIV_H
#define NNL2_DIV_H

/** @brief
 * Performs element-wise division of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the quotient of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy. Checks for division by zero.
 *
 ** @param dividend
 * Pointer to the dividend tensor
 *
 ** @param divisor
 * Pointer to the divisor tensor
 *
 ** @return 
 * Pointer to a new tensor with the division result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure or division by zero
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_naive_div(const Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_dividend, dtype_divisor);

    // Create an output tensor with the same shape and winning data type
    Tensor* quotient = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);
    
    if (quotient == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return quotient;
    }
    
    if (dtype_dividend == dtype_divisor) {
        // Handling the case if the data types match
        
        switch (dtype_dividend) {
            case FLOAT64: {
                volatile double* data_dividend = (double*)dividend->data;
                volatile double* data_divisor = (double*)divisor->data;
                volatile double* data_quotient = (double*)quotient->data;
            
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_dividend = (float*)dividend->data;
                volatile float* data_divisor = (float*)divisor->data;
                volatile float* data_quotient = (float*)quotient->data;
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0.0f) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_dividend = (int32_t*)dividend->data;
                volatile int32_t* data_divisor = (int32_t*)divisor->data;
                volatile int32_t* data_quotient = (int32_t*)quotient->data;
        
                // Element-wise division with zero check
                for (size_t i = 0; i < len; i++) {
                    if (data_divisor[i] == 0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    data_quotient[i] = data_dividend[i] / data_divisor[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_dividend);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_quotient = (double*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    double divisor_val = nnl2_convert_to_float64(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float64(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_quotient = (float*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    float divisor_val = nnl2_convert_to_float32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0.0f) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_float32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_quotient = (int32_t*)quotient->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_dividend = (char*)dividend->data + i * get_dtype_size(dtype_dividend);
                    void* elem_divisor = (char*)divisor->data + i * get_dtype_size(dtype_divisor);
                    
                    int32_t divisor_val = nnl2_convert_to_int32(elem_divisor, dtype_divisor);
                    if (divisor_val == 0) {
                        NNL2_ERROR("Division by zero at index %zu\n", i);
                        nnl2_free_tensor(quotient);
                        return NULL;
                    }
                    
                    data_quotient[i] = nnl2_convert_to_int32(elem_dividend, dtype_dividend) / divisor_val;
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                nnl2_free_tensor(quotient);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return quotient;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * division operation
 */
#define NNL2_DIV_PARALLEL_THREASHOLD 1000000

/** @brief
 * Task structure for parallel division operation
 */
typedef struct {
    const void* dividend_data;    /**< Pointer to dividend tensor data */
    const void* divisor_data;     /**< Pointer to divisor tensor data */
    void* result_data;            /**< Pointer to result tensor data */
    size_t start;                 /**< Start index for this thread */
    size_t end;                   /**< End index for this thread */
    TensorType dtype_dividend;    /**< Data type of dividend tensor */
    TensorType dtype_divisor;     /**< Data type of divisor tensor */
    TensorType result_dtype;      /**< Data type of result tensor */
} div_ptask;

/** @brief 
 * Worker function for parallel division for same data types
 * 
 * @param arg 
 * Pointer to div_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pdiv_same_type(void* arg);

/** @brief 
 * Worker function for parallel division for mixed data types
 * 
 * @param arg 
 * Pointer to div_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pdiv_mixed_types(void* arg);

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * SIMD-optimized worker function for parallel division for same float64 data types
 * 
 * @param arg 
 * Pointer to div_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pdiv_simd_float64(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel division for same float32 data types
 * 
 * @param arg 
 * Pointer to div_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pdiv_simd_float32(void* arg);

#endif

/** @brief
 * Parallel implementation of tensor division using pthreads
 *
 ** @param dividend
 * Pointer to the dividend tensor
 *
 ** @param divisor
 * Pointer to the divisor tensor
 *
 ** @return 
 * Pointer to a new tensor with the division result
 */
Tensor* nnl2_own_div(const Tensor* dividend, const Tensor* divisor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(dividend->shape, dividend->rank);
    
    TensorType dtype_dividend = dividend->dtype;
    TensorType dtype_divisor = divisor->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_dividend, dtype_divisor);

    // Create an output tensor with the same shape and data type
    Tensor* quotient = nnl2_empty(dividend->shape, dividend->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return quotient;
    
    // Use naive implementation for small tensors
    if(len < NNL2_DIV_PARALLEL_THREASHOLD) {
        quotient = nnl2_naive_div(dividend, divisor);
        if(quotient == NULL) {
            NNL2_ERROR("Failed to divide");
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return quotient;
    }
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[NNL2_NUM_THREADS];
    div_ptask tasks[NNL2_NUM_THREADS];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = len / NNL2_NUM_THREADS;
    size_t remainder = len % NNL2_NUM_THREADS;
    
    bool use_simd = false;
    
    #ifdef NNL2_AVX256_AVAILABLE
    if(dtype_dividend == dtype_divisor && (dtype_dividend == FLOAT64 || dtype_dividend == FLOAT32)) {
        bool aligned_dividend = NNL2_IS_ALIGNED(dividend->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_divisor = NNL2_IS_ALIGNED(divisor->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_result = NNL2_IS_ALIGNED(quotient->data, NNL2_TENSOR_ALIGNMENT_32);
        use_simd = aligned_dividend && aligned_divisor && aligned_result;
    }
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].dividend_data = dividend->data;
        tasks[i].divisor_data = divisor->data;
        tasks[i].result_data = quotient->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_dividend = dtype_dividend;
        tasks[i].dtype_divisor = dtype_divisor;
        tasks[i].result_dtype = winner_in_the_type_hierarchy;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
            if(use_simd && dtype_dividend == dtype_divisor) {
                switch(dtype_dividend) {
                    case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_pdiv_simd_float64, &tasks[i]); break;
                    case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_pdiv_simd_float32, &tasks[i]); break;
                    
                    default: {
                        status = pthread_create(&threads[i], NULL, nnl2_own_pdiv_same_type, &tasks[i]);
                        break;
                    }
                }
            } else 
        #endif
        {
            if(dtype_dividend == dtype_divisor) {
                status = pthread_create(&threads[i], NULL, nnl2_own_pdiv_same_type, &tasks[i]);
            } else {
                status = pthread_create(&threads[i], NULL, nnl2_own_pdiv_mixed_types, &tasks[i]);
            }
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_div");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            
            nnl2_free_tensor(quotient);
            return NULL;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_div");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return quotient;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pdiv_same_type
 **/
void* nnl2_own_pdiv_same_type(void* arg) {
    div_ptask* task = (div_ptask*)arg;
    
    switch(task->dtype_dividend) {
        case FLOAT64: {
            volatile double* data_dividend = (double*)task->dividend_data;
            volatile double* data_divisor = (double*)task->divisor_data;
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                if (data_divisor[i] == 0.0) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL; // This will cause the whole operation to fail
                }
				
                data_result[i] = data_dividend[i] / data_divisor[i];
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_dividend = (float*)task->dividend_data;
            volatile float* data_divisor = (float*)task->divisor_data;
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                if (data_divisor[i] == 0.0f) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL; // This will cause the whole operation to fail
                }
				
                data_result[i] = data_dividend[i] / data_divisor[i];
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_dividend = (int32_t*)task->dividend_data;
            volatile int32_t* data_divisor = (int32_t*)task->divisor_data;
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                if (data_divisor[i] == 0) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL; // This will cause the whole operation to fail
                }
				
                data_result[i] = data_dividend[i] / data_divisor[i];
            }
            
            break;
        }
        
        default: {
            // Type error should be handled in main function
            break;
        }
    }
    
    return NULL;
}

#ifdef NNL2_AVX256_AVAILABLE

/** @brief
 * SIMD-optimized worker function for float64 division
 *
 ** @see nnl2_own_pdiv_simd_float64
 **/
void* nnl2_own_pdiv_simd_float64(void* arg) {
    div_ptask* task = (div_ptask*)arg;
    
    double* data_dividend = (double*)task->dividend_data;
    double* data_divisor = (double*)task->divisor_data;
    double* data_result = (double*)task->result_data;
    
    size_t i = task->start;
    
    // Check for zeros in the divisor first (scalar check for safety)
    for(size_t j = task->start; j < task->end; j++) {
        if(data_divisor[j] == 0.0) {
            NNL2_ERROR("Division by zero at index %zu\n", j);
            return NULL;
        }
    }
    
    // Process 4 elements at a time using AVX
    for(; i + 3 < task->end; i += 4) {
        __m256d v_dividend = _mm256_load_pd(&data_dividend[i]);        // Load 4 doubles
        __m256d v_divisor = _mm256_load_pd(&data_divisor[i]);          // Load 4 doubles
        __m256d v_result = _mm256_div_pd(v_dividend, v_divisor);       // Vector division
        _mm256_store_pd(&data_result[i], v_result);                    // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_dividend[i] / data_divisor[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for float32 division
 *
 ** @see nnl2_own_pdiv_simd_float32
 **/
void* nnl2_own_pdiv_simd_float32(void* arg) {
    div_ptask* task = (div_ptask*)arg;
    
    float* data_dividend = (float*)task->dividend_data;
    float* data_divisor = (float*)task->divisor_data;
    float* data_result = (float*)task->result_data;
    
    size_t i = task->start;
    
    // Check for zeros in the divisor first (scalar check for safety)
    for(size_t j = task->start; j < task->end; j++) {
        if(data_divisor[j] == 0.0f) {
            NNL2_ERROR("Division by zero at index %zu\n", j);
            return NULL;
        }
    }
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256 v_dividend = _mm256_load_ps(&data_dividend[i]);        // Load 8 floats
        __m256 v_divisor = _mm256_load_ps(&data_divisor[i]);          // Load 8 floats
        __m256 v_result = _mm256_div_ps(v_dividend, v_divisor);       // Vector division
        _mm256_store_ps(&data_result[i], v_result);                   // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_dividend[i] / data_divisor[i];
    }
    
    return NULL;
}

#endif

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pdiv_mixed_types
 **/
void* nnl2_own_pdiv_mixed_types(void* arg) {
    div_ptask* task = (div_ptask*)arg;
    
    switch(task->result_dtype) {
        case FLOAT64: {
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_dividend = (char*)task->dividend_data + i * get_dtype_size(task->dtype_dividend);
                void* elem_divisor = (char*)task->divisor_data + i * get_dtype_size(task->dtype_divisor);
                
                double divisor_val = nnl2_convert_to_float64(elem_divisor, task->dtype_divisor);
                if (divisor_val == 0.0) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL;
                }
                
                data_result[i] = nnl2_convert_to_float64(elem_dividend, task->dtype_dividend) / divisor_val;
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_dividend = (char*)task->dividend_data + i * get_dtype_size(task->dtype_dividend);
                void* elem_divisor = (char*)task->divisor_data + i * get_dtype_size(task->dtype_divisor);
                
                float divisor_val = nnl2_convert_to_float32(elem_divisor, task->dtype_divisor);
                if (divisor_val == 0.0f) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL;
                }
                
                data_result[i] = nnl2_convert_to_float32(elem_dividend, task->dtype_dividend) / divisor_val;
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_dividend = (char*)task->dividend_data + i * get_dtype_size(task->dtype_dividend);
                void* elem_divisor = (char*)task->divisor_data + i * get_dtype_size(task->dtype_divisor);
                
                int32_t divisor_val = nnl2_convert_to_int32(elem_divisor, task->dtype_divisor);
                if (divisor_val == 0) {
                    NNL2_ERROR("Division by zero at index %zu\n", i);
                    return NULL;
                }
                
                data_result[i] = nnl2_convert_to_int32(elem_dividend, task->dtype_dividend) / divisor_val;
            }
            
            break;
        }
        
        default: {
            // Type error should be handled in main function
            break;
        }
    }
    
    return NULL;
}

#endif

/** 
 * @ingroup backend_system
 * @brief Backend implementations for division operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_div: Basic reference implementation
 *  - nnl2_own_div: nnl2 Own hyper-optimized implementation
 * 
 * @see nnl2_naive_div
 * @see nnl2_own_div
 */
Implementation div_backends[] = {
    REGISTER_BACKEND(nnl2_naive_div, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_div, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for division operation
 * @ingroup backend_system 
 */
divfn nnl2_div;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(div);

/** 
 * @brief Sets the backend for division operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_div_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(div_backends, div, backend_name, CURRENT_BACKEND(div));
}

/** 
 * @brief Gets the name of the active backend for division operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_div_backend() {
    return CURRENT_BACKEND(div);
}

/** 
 * @brief Function declaration for getting all `div` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(div);

/**
 * @brief Function declaration for getting the number of all `div` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(div);

#endif /** NNL2_DIV_H **/