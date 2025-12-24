#ifndef NNL2_MUL_H
#define NNL2_MUL_H

/** @brief
 * Performs element-wise multiplication of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the nnl2_product of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param multiplicand
 * Pointer to the multiplicand tensor
 *
 ** @param multiplier
 * Pointer to the multiplier tensor
 *
 ** @return 
 * Pointer to a new tensor with the multiplication result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
nnl2_tensor* nnl2_naive_mul(const nnl2_tensor* multiplicand, const nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = nnl2_product(multiplicand->shape, multiplicand->rank);
    
    nnl2_tensor_type dtype_multiplicand = multiplicand->dtype;
    nnl2_tensor_type dtype_multiplier = multiplier->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_multiplicand, dtype_multiplier);

    // Create an output tensor with the same shape and winning data type
    nnl2_tensor* nnl2_product = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);
    
    if (nnl2_product == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return nnl2_product;
    }
    
    if (dtype_multiplicand == dtype_multiplier) {
        // Handling the case if the data types match
        
        switch (dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                volatile double* data_product = (double*)nnl2_product->data;
            
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                volatile float* data_product = (float*)nnl2_product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
			
			case INT64: {
                volatile int64_t* data_multiplicand = (int64_t*)multiplicand->data;
                volatile int64_t* data_multiplier = (int64_t*)multiplier->data;
                volatile int64_t* data_product = (int64_t*)nnl2_product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                volatile int32_t* data_product = (int32_t*)nnl2_product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_product = (double*)nnl2_product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float64(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float64(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_product = (float*)nnl2_product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
			
			case INT64: {
                volatile int64_t* data_product = (int64_t*)nnl2_product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_int64(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_int64(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_product = (int32_t*)nnl2_product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_int32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_int32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nnl2_product;
}

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of the
 * multiplication operation
 */
#define NNL2_MUL_PARALLEL_THREASHOLD 1000000

/** @brief 
 * Worker function for parallel multiplication for same data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_same_type(void* arg);

/** @brief 
 * Worker function for parallel multiplication for mixed data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_mixed_types(void* arg);

#ifdef NNL2_AVX256_AVAILABLE

/** @brief 
 * SIMD-optimized worker function for parallel multiplication for same float64 data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_simd_float64(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel multiplication for same float32 data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_simd_float32(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel multiplication for same int32 data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_simd_int32(void* arg);

/** @brief 
 * SIMD-optimized worker function for parallel multiplication for same int64 data types
 * 
 * @param arg 
 * Pointer to mul_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_pmul_simd_int64(void* arg);

#endif

/** @brief
 * Parallel implementation of tensor multiplication using pthreads
 *
 ** @param multiplicand
 * Pointer to the multiplicand tensor
 *
 ** @param multiplier
 * Pointer to the multiplier tensor
 *
 ** @return 
 * Pointer to a new tensor with the multiplication result
 */
nnl2_tensor* nnl2_own_mul(const nnl2_tensor* multiplicand, const nnl2_tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = nnl2_product(multiplicand->shape, multiplicand->rank);
    
    nnl2_tensor_type dtype_multiplicand = multiplicand->dtype;
    nnl2_tensor_type dtype_multiplier = multiplier->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    nnl2_tensor_type winner_in_the_type_hierarchy = MAX(dtype_multiplicand, dtype_multiplier);

    // Create an output tensor with the same shape and data type
    nnl2_tensor* nnl2_product = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);
    
    if(len == 0) return nnl2_product;
    
    // Use naive implementation for small tensors
    if(len < NNL2_MUL_PARALLEL_THREASHOLD) {
        nnl2_product = nnl2_naive_mul(multiplicand, multiplier);
        if(nnl2_product == NULL) {
            NNL2_ERROR("Failed to multiply");
        }
        
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        
        return nnl2_product;
    }
    
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[NNL2_NUM_THREADS];
    mul_ptask tasks[NNL2_NUM_THREADS];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = len / NNL2_NUM_THREADS;
    size_t remainder = len % NNL2_NUM_THREADS;
    
    bool use_simd = false;
    
    #ifdef NNL2_AVX256_AVAILABLE
    if(dtype_multiplicand == dtype_multiplier) {
        bool aligned_multiplicand = NNL2_IS_ALIGNED(multiplicand->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_multiplier = NNL2_IS_ALIGNED(multiplier->data, NNL2_TENSOR_ALIGNMENT_32);
        bool aligned_result = NNL2_IS_ALIGNED(nnl2_product->data, NNL2_TENSOR_ALIGNMENT_32);
        use_simd = aligned_multiplicand && aligned_multiplier && aligned_result;
    }
    #endif
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].multiplicand_data = multiplicand->data;
        tasks[i].multiplier_data = multiplier->data;
        tasks[i].result_data = nnl2_product->data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        tasks[i].dtype_multiplicand = dtype_multiplicand;
        tasks[i].dtype_multiplier = dtype_multiplier;
        tasks[i].result_dtype = winner_in_the_type_hierarchy;
        
        // Create thread to process the assigned chunk
        int status;
        
        #ifdef NNL2_AVX256_AVAILABLE
            if(use_simd && dtype_multiplicand == dtype_multiplier) {
                switch(dtype_multiplicand) {
                    case FLOAT64: status = pthread_create(&threads[i], NULL, nnl2_own_pmul_simd_float64, &tasks[i]); break;
                    case FLOAT32: status = pthread_create(&threads[i], NULL, nnl2_own_pmul_simd_float32, &tasks[i]); break;
                    case INT32:   status = pthread_create(&threads[i], NULL, nnl2_own_pmul_simd_int32, &tasks[i]);   break;
                    case INT64:   status = pthread_create(&threads[i], NULL, nnl2_own_pmul_simd_int64, &tasks[i]);   break;
                    
                    default: {
                        status = pthread_create(&threads[i], NULL, nnl2_own_pmul_same_type, &tasks[i]);
                        break;
                    }
                }
            } else 
        #endif
        {
            if(dtype_multiplicand == dtype_multiplier) {
                status = pthread_create(&threads[i], NULL, nnl2_own_pmul_same_type, &tasks[i]);
            } else {
                status = pthread_create(&threads[i], NULL, nnl2_own_pmul_mixed_types, &tasks[i]);
            }
        }
        
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_mul");
            // Clean up already created threads
            for(size_t j = 0; j < i; j++) {
                pthread_join(threads[j], NULL);
            }
            
            nnl2_free_tensor(nnl2_product);
            return NULL;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < NNL2_NUM_THREADS; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_mul");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nnl2_product;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pmul_same_type
 **/
void* nnl2_own_pmul_same_type(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    switch(task->dtype_multiplicand) {
        case FLOAT64: {
            volatile double* data_multiplicand = (double*)task->multiplicand_data;
            volatile double* data_multiplier = (double*)task->multiplier_data;
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_multiplicand = (float*)task->multiplicand_data;
            volatile float* data_multiplier = (float*)task->multiplier_data;
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
		
		case INT64: {
            volatile int64_t* data_multiplicand = (int64_t*)task->multiplicand_data;
            volatile int64_t* data_multiplier = (int64_t*)task->multiplier_data;
            volatile int64_t* data_result = (int64_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_multiplicand[i] * data_multiplier[i];
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_multiplicand = (int32_t*)task->multiplicand_data;
            volatile int32_t* data_multiplier = (int32_t*)task->multiplier_data;
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                data_result[i] = data_multiplicand[i] * data_multiplier[i];
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
 * SIMD-optimized worker function for float64 multiplication
 *
 ** @see nnl2_own_pmul_simd_float64
 **/
void* nnl2_own_pmul_simd_float64(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    double* data_multiplicand = (double*)task->multiplicand_data;
    double* data_multiplier = (double*)task->multiplier_data;
    double* data_result = (double*)task->result_data;
    
    size_t i = task->start;
    
    // Process 4 elements at a time using AVX
    for(; i + 3 < task->end; i += 4) {
        __m256d v_multiplicand = _mm256_load_pd(&data_multiplicand[i]);        // Load 4 doubles
        __m256d v_multiplier = _mm256_load_pd(&data_multiplier[i]);            // Load 4 doubles
        __m256d v_result = _mm256_mul_pd(v_multiplicand, v_multiplier);        // Vector multiplication
        _mm256_store_pd(&data_result[i], v_result);                            // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_multiplicand[i] * data_multiplier[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for float32 multiplication
 *
 ** @see nnl2_own_pmul_simd_float32
 **/
void* nnl2_own_pmul_simd_float32(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    float* data_multiplicand = (float*)task->multiplicand_data;
    float* data_multiplier = (float*)task->multiplier_data;
    float* data_result = (float*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256 v_multiplicand = _mm256_load_ps(&data_multiplicand[i]);        // Load 8 floats
        __m256 v_multiplier = _mm256_load_ps(&data_multiplier[i]);            // Load 8 floats
        __m256 v_result = _mm256_mul_ps(v_multiplicand, v_multiplier);        // Vector multiplication
        _mm256_store_ps(&data_result[i], v_result);                           // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_multiplicand[i] * data_multiplier[i];
    }
    
    return NULL;
}

/** @brief
 * optimized worker function for int64 multiplication
 *
 ** @see nnl2_own_pmul_simd_int64
 **/
void* nnl2_own_pmul_simd_int64(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    int64_t* data_multiplicand = (int64_t*)task->multiplicand_data;
    int64_t* data_multiplier = (int64_t*)task->multiplier_data;
    int64_t* data_result = (int64_t*)task->result_data;
    
    size_t i = task->start;
    
    for(; i + 3 < task->end; i += 4) {
        data_result[i] = data_multiplicand[i] * data_multiplier[i];
        data_result[i+1] = data_multiplicand[i+1] * data_multiplier[i+1];
        data_result[i+2] = data_multiplicand[i+2] * data_multiplier[i+2];
        data_result[i+3] = data_multiplicand[i+3] * data_multiplier[i+3];
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_multiplicand[i] * data_multiplier[i];
    }
    
    return NULL;
}

/** @brief
 * SIMD-optimized worker function for int32 multiplication
 *
 ** @see nnl2_own_pmul_simd_int32
 **/
void* nnl2_own_pmul_simd_int32(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    int32_t* data_multiplicand = (int32_t*)task->multiplicand_data;
    int32_t* data_multiplier = (int32_t*)task->multiplier_data;
    int32_t* data_result = (int32_t*)task->result_data;
    
    size_t i = task->start;
    
    // Process 8 elements at a time using AVX
    for(; i + 7 < task->end; i += 8) {
        __m256i v_multiplicand = _mm256_load_si256((__m256i*)&data_multiplicand[i]);  // Load 8 int32
        __m256i v_multiplier = _mm256_load_si256((__m256i*)&data_multiplier[i]);      // Load 8 int32
        
        __m256i v_mul_lo = _mm256_mullo_epi16(v_multiplicand, v_multiplier);          // Multiply low 16 bits
        __m256i v_mul_hi = _mm256_mulhi_epi16(v_multiplicand, v_multiplier);          // Multiply high 16 bits
        
        // Combine results 
        __m256i v_result = _mm256_or_si256(v_mul_lo, _mm256_slli_epi32(v_mul_hi, 16));
        
        _mm256_store_si256((__m256i*)&data_result[i], v_result);                      // Store result
    }
    
    // Process remainder elements
    for(; i < task->end; i++) {
        data_result[i] = data_multiplicand[i] * data_multiplier[i];
    }
    
    return NULL;
}

#endif

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_pmul_mixed_types
 **/
void* nnl2_own_pmul_mixed_types(void* arg) {
    mul_ptask* task = (mul_ptask*)arg;
    
    switch(task->result_dtype) {
        case FLOAT64: {
            volatile double* data_result = (double*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_multiplicand = (char*)task->multiplicand_data + i * get_dtype_size(task->dtype_multiplicand);
                void* elem_multiplier = (char*)task->multiplier_data + i * get_dtype_size(task->dtype_multiplier);
                
                data_result[i] = nnl2_convert_to_float64(elem_multiplicand, task->dtype_multiplicand) * nnl2_convert_to_float64(elem_multiplier, task->dtype_multiplier);
            }
            
            break;
        }
        
        case FLOAT32: {
            volatile float* data_result = (float*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_multiplicand = (char*)task->multiplicand_data + i * get_dtype_size(task->dtype_multiplicand);
                void* elem_multiplier = (char*)task->multiplier_data + i * get_dtype_size(task->dtype_multiplier);
                
                data_result[i] = nnl2_convert_to_float32(elem_multiplicand, task->dtype_multiplicand) * nnl2_convert_to_float32(elem_multiplier, task->dtype_multiplier);
            }
            
            break;
        }
		
		case INT64: {
            volatile int64_t* data_result = (int64_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_multiplicand = (char*)task->multiplicand_data + i * get_dtype_size(task->dtype_multiplicand);
                void* elem_multiplier = (char*)task->multiplier_data + i * get_dtype_size(task->dtype_multiplier);
                
                data_result[i] = nnl2_convert_to_int64(elem_multiplicand, task->dtype_multiplicand) * nnl2_convert_to_int64(elem_multiplier, task->dtype_multiplier);
            }
            
            break;
        }
        
        case INT32: {
            volatile int32_t* data_result = (int32_t*)task->result_data;
            
            for(size_t i = task->start; i < task->end; i++) {
                void* elem_multiplicand = (char*)task->multiplicand_data + i * get_dtype_size(task->dtype_multiplicand);
                void* elem_multiplier = (char*)task->multiplier_data + i * get_dtype_size(task->dtype_multiplier);
                
                data_result[i] = nnl2_convert_to_int32(elem_multiplicand, task->dtype_multiplicand) * nnl2_convert_to_int32(elem_multiplier, task->dtype_multiplier);
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
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mul: Basic reference implementation
 *  - nnl2_own_mul: nnl2 Own hyper-optimized implementation
 * 
 * @see nnl2_naive_mul
 * @see nnl2_own_mul
 */
nnl2_runtime_implementation mul_backends[] = {
    REGISTER_BACKEND(nnl2_naive_mul, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_mul, nnl2_own, NNL2_OWN_NAME),
    #endif
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulfn mul;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(mul);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mul_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mul_backends, mul, backend_name, CURRENT_BACKEND(mul));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mul_backend() {
    return CURRENT_BACKEND(mul);
}

/** 
 * @brief Function declaration for getting all `mul` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mul);

/**
 * @brief Function declaration for getting the number of all `mul` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mul);

#endif /** NNL2_MUL_H **/
