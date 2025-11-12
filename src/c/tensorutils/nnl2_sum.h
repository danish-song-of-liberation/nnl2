#ifndef NNL2_SUM_H
#define NNL2_SUM_H

/** @brief
 * Computes the sum of all elements in a tensor without axis reduction
 * 
 ** @param tensor 
 * Pointer to the input tensor structure
 *
 ** @param result 
 * Pointer to the memory where the sum result will be stored
 */
void naive_sum_without_axis(Tensor* tensor, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If tensor if empty then return empty result
		
	switch(tensor->dtype) {
		case FLOAT64: {
            double* cast_data = (double*)tensor->data;
            double acc = 0.0;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((double*)result) = acc; // Store result in output pointer
            break; 
        }
	
		case FLOAT32: {
            float* cast_data = (float*)tensor->data;
            float acc = 0.0f;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((float*)result) = acc; 
            break;
        }
			
		case INT32: {
            int32_t* cast_data = (int32_t*)tensor->data;
            int32_t acc = 0;
            for (size_t it = 0; it < total_elems; it++) acc += cast_data[it];
            *((int32_t*)result) = acc; 
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
 * Task structure for parallel sum without axis operations
 */
typedef struct {
    void* src_data;           /**< Pointer to source data */
    size_t start_idx;         /**< Start index for this thread */
    size_t end_idx;           /**< End index for this thread */
    TensorType dtype;         /**< Data type of the tensor */
    bool aligned;             /**< Whether memory is aligned */
    union {
        double float64_acc;
        float float32_acc;
        int32_t int32_acc;
    } accumulator;            /**< Thread-local accumulator */
} sum_ptask;

/** @brief
 * Threshold for enabling parallel execution of sum operation
 */
#define NNL2_SUM_PARALLEL_THRESHOLD 100000

/** @brief
 * Worker function for parallel double precision sum
 */
void* nnl2_own_psum_float64(void* arg);

/** @brief
 * Worker function for parallel single precision sum
 */
void* nnl2_own_psum_float32(void* arg);

/** @brief
 * Worker function for parallel integer sum
 */
void* nnl2_own_psum_int32(void* arg);

/** @brief
 * High-performance parallel implementation of tensor sum without axis
 * 
 ** @param tensor 
 * Pointer to the input tensor structure
 *
 ** @param result 
 * Pointer to the memory where the sum result will be stored
 *
 ** @details
 * Uses multi-threading with pthread and AVX256 vectorization for
 * maximum performance on modern CPU architectures.
 */
void nnl2_own_sum_without_axis(Tensor* tensor, void* result) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;
    
    // Fallback to naive implementation for small tensors
    if(total_elems < NNL2_SUM_PARALLEL_THRESHOLD) {
        naive_sum_without_axis(tensor, result);
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return;
    }
    
    bool is_aligned = NNL2_IS_ALIGNED(tensor->data, NNL2_TENSOR_ALIGNMENT_32);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_sum_without_axis, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    sum_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes with load balancing
    size_t chunk = total_elems / num_threads;
    size_t remainder = total_elems % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].dtype = tensor->dtype;
        tasks[i].aligned = is_aligned;
        tasks[i].src_data = tensor->data;
        
        // Initialize accumulators
        switch(tensor->dtype) {
            case FLOAT64: tasks[i].accumulator.float64_acc = 0.0; break;
            case FLOAT32: tasks[i].accumulator.float32_acc = 0.0f; break;
            case INT32:   tasks[i].accumulator.int32_acc = 0; break;
            default: break;
        }
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        // Select appropriate worker function
        void* (*worker_func)(void*) = NULL;
        switch(tensor->dtype) {
            case FLOAT64: worker_func = nnl2_own_psum_float64; break;
            case FLOAT32: worker_func = nnl2_own_psum_float32; break;
            case INT32:   worker_func = nnl2_own_psum_int32;   break;
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                    NNL2_FUNC_EXIT();
                #endif
                return;
            }
        }
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sum_without_axis");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete and accumulate results
    switch(tensor->dtype) {
        case FLOAT64: {
            double total = 0.0;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total += tasks[i].accumulator.float64_acc;
            }
            *((double*)result) = total;
            break;
        }
        case FLOAT32: {
            float total = 0.0f;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total += tasks[i].accumulator.float32_acc;
            }
            *((float*)result) = total;
            break;
        }
        case INT32: {
            int32_t total = 0;
            for (size_t i = 0; i < num_threads; i++) {
                pthread_join(threads[i], NULL);
                total += tasks[i].accumulator.int32_acc;
            }
            *((int32_t*)result) = total;
            break;
        }
        default: break;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

// Worker function implementations with AVX256

void* nnl2_own_psum_float64(void* arg) {
    sum_ptask* task = (sum_ptask*)arg;
    double* data = (double*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    double sum = 0.0;
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
    if(task->aligned && (end - start) >= 4) {
        __m256d v_sum = _mm256_setzero_pd();
        
        // AVX256 processing (4 elements per iteration)
        for(; i + 3 < end; i += 4) {
            __m256d v_data = _mm256_load_pd(&data[i]);
            v_sum = _mm256_add_pd(v_sum, v_data);
        }
        
        // Horizontal sum of AVX vector
        double temp[4] __attribute__((aligned(32)));
        _mm256_store_pd(temp, v_sum);
        sum = temp[0] + temp[1] + temp[2] + temp[3];
    }
    #endif
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        sum += data[i];
    }
    
    task->accumulator.float64_acc = sum;
    return NULL;
}

void* nnl2_own_psum_float32(void* arg) {
    sum_ptask* task = (sum_ptask*)arg;
    float* data = (float*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    float sum = 0.0f;
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
    if(task->aligned && (end - start) >= 8) {
        __m256 v_sum = _mm256_setzero_ps();
        
        // AVX256 processing (8 elements per iteration)
        for(; i + 7 < end; i += 8) {
            __m256 v_data = _mm256_load_ps(&data[i]);
            v_sum = _mm256_add_ps(v_sum, v_data);
        }
        
        // Horizontal sum of AVX vector
        float temp[8] __attribute__((aligned(32)));
        _mm256_store_ps(temp, v_sum);
        sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
    #endif
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        sum += data[i];
    }
    
    task->accumulator.float32_acc = sum;
    return NULL;
}

void* nnl2_own_psum_int32(void* arg) {
    sum_ptask* task = (sum_ptask*)arg;
    int32_t* data = (int32_t*)task->src_data;
    size_t start = task->start_idx;
    size_t end = task->end_idx;
    
    int32_t sum = 0;
    size_t i = start;
    
    #if defined(NNL2_AVX256_AVAILABLE)
    if(task->aligned && (end - start) >= 8) {
        __m256i v_sum = _mm256_setzero_si256();
        
        // AVX256 processing (8 elements per iteration)
        for(; i + 7 < end; i += 8) {
            __m256i v_data = _mm256_load_si256((__m256i*)&data[i]);
            v_sum = _mm256_add_epi32(v_sum, v_data);
        }
        
        // Horizontal sum of AVX vector
        int32_t temp[8] __attribute__((aligned(32)));
        _mm256_store_si256((__m256i*)temp, v_sum);
        sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
    }
    #endif
    
    // Scalar processing for remainder
    for(; i < end; i++) {
        sum += data[i];
    }
    
    task->accumulator.int32_acc = sum;
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum without axis operation
 * @details
 * Array follows the common backend registration pattern for element-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation
 * 
 * @see nnl2_naive
 * @see naive_sum_without_axis
 */
Implementation sum_without_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_without_axis, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_sum_without_axis, nnl2_own, NNL2_OWN_NAME),
    #endif
};	

/**
 * @brief Function pointer for sum without axis operation
 * @ingroup backend_system 
 */
sumwithoutaxisfn nnl2_sum_without_axis;

/** 
 * @brief Makes the sum without axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_without_axis);

/** 
 * @brief Sets the backend for sum without axis operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for summation
 * @see ESET_BACKEND_BY_NAME
 */
void set_sum_without_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_without_axis_backends, nnl2_sum_without_axis, backend_name, CURRENT_BACKEND(sum_without_axis));
}

/** 
 * @brief Gets the name of the active backend for sum without axis operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sum_without_axis_backend() {
    return CURRENT_BACKEND(sum_without_axis);
}

/** 
 * @brief Function declaration for getting all available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sum_without_axis);

/**
 * @brief Function declaration for getting the number of available sum without axis backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sum_without_axis);

/** @brief
 * Calculates the source index in the tensor for the axis-wise summation operation
 *
 ** @param tensor
 * Original tensor
 *
 ** @param result
 * Result tensor
 *
 ** @param axis
 * The axis along which the summation is performed
 *
 ** @param result_index
 * Index in the resulting tensor
 * 
 ** @param axis_index
 * Position along the summation axis
 * 
 ** @return 
 * Calculated index in the original tensor
 */
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index);

/** @brief
 * Performs axis summation for a FLOAT64 tensor
 *
 ** @param tensor
 * Original tensor with double (float64) type
 *
 ** @param result
 * The preallocated resulting tensor
 *
 ** @param axis
 * Sum axis (0-based)
 *
 ** @param result_numel
 * The number of elements in the result
 *
 ** @param elements_along_axis
 * The number of elements along the summation axis
 *
 ** @return
 * Tensor* Pointer to the resulting tensor
 */
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the float32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

/** @brief
 * The documentation is identical to the 
 * nnl2_naive_sum_with_axis_float64 but with the int32 type
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis);

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Worker function for parallel axis sum
 */
void* nnl2_own_paxis_sum(void* arg);

/** @brief
 * High-performance parallel implementation of axis-wise tensor sum
 */
Tensor* nnl2_own_sum_with_axis(Tensor* tensor, int axis, bool keepdim);

#endif

/** @brief
 * Computes the sum of tensor elements along the specified axis
 *
 ** @param tensor
 * Input tensor to perform summation on
 *
 ** @param axis
 * Axis along which to sum (0-based index)
 *
 ** @param keepdim
 * If true, retains reduced dimensions with size 1
 *
 ** @return
 * Tensor* New tensor with reduced rank containing the sums
 */
Tensor* naive_sum_with_axis(Tensor* tensor, int axis, bool keepdim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Validate axis parameter
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (axis < 0 || axis >= tensor->rank) {
			NNL2_ERROR("Invalid axis %d for tensor of rank %d", axis, tensor->rank);
			return NULL;
		}
	#endif
    
    // Calculate result rank
    int result_rank = keepdim ? tensor->rank : tensor->rank - 1;
    
	// Allocate memory for result shape
    int* result_shape = malloc(result_rank * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result_shape == NULL) {
			NNL2_ERROR("Failed to allocate memory");
			return NULL;
		}
	#endif
	
	// Construct result shape
    if (keepdim) {
        // Keep all dimensions, set summed axis to 1
        for (int i = 0; i < tensor->rank; i++) {
            result_shape[i] = (i == axis) ? 1 : tensor->shape[i];
        }
    } else {
        // Exclude the summation axis
        int j = 0;
        for (int i = 0; i < tensor->rank; i++) {
            if (i != axis) {
                result_shape[j++] = tensor->shape[i];
            }
        }
    }

    // Create empty result tensor with appropriate shape and dtype
    Tensor* result = nnl2_empty(result_shape, result_rank, tensor->dtype);
    free(result_shape);
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(result == NULL) {
			NNL2_ERROR("Failed to create result tensor");
			return NULL;
		}
	#endif
	
    size_t result_numel = product(result->shape, result->rank);
    int elements_along_axis = tensor->shape[axis];
    
    // Use optimized implementation for large tensors
    #ifdef NNL2_PTHREAD_AVAILABLE
		if (result_numel * elements_along_axis >= NNL2_SUM_PARALLEL_THRESHOLD) {
			return nnl2_own_sum_with_axis(tensor, axis, keepdim);
		}
    #endif
	
	// Select appropriate summation function based on data type
	Tensor* (*sum_func)(Tensor*, Tensor*, int, size_t, int) = NULL;
	
	switch(tensor->dtype) {
        case FLOAT32: {
            sum_func = nnl2_naive_sum_with_axis_float32;
            break;
		}
		
        case FLOAT64: {
            sum_func = nnl2_naive_sum_with_axis_float64;
            break;
		}
		
        case INT32: {
            sum_func = nnl2_naive_sum_with_axis_int32;
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
    
	// Perform the actual summation
    return sum_func(tensor, result, axis, result_numel, elements_along_axis);
}

#ifdef NNL2_PTHREAD_AVAILABLE

Tensor* nnl2_own_sum_with_axis(Tensor* tensor, int axis, bool keepdim) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate result rank and shape
    int result_rank = keepdim ? tensor->rank : tensor->rank - 1;
    int* result_shape = malloc(result_rank * sizeof(int));
    if(result_shape == NULL) {
        NNL2_ERROR("Failed to allocate memory");
        return NULL;
    }
    
    // Construct result shape
    if (keepdim) {
        for (int i = 0; i < tensor->rank; i++) {
            result_shape[i] = (i == axis) ? 1 : tensor->shape[i];
        }
    } else {
        int j = 0;
        for (int i = 0; i < tensor->rank; i++) {
            if (i != axis) {
                result_shape[j++] = tensor->shape[i];
            }
        }
    }

    Tensor* result = nnl2_empty(result_shape, result_rank, tensor->dtype);
    free(result_shape);
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor");
        return NULL;
    }
    
    size_t result_numel = product(result->shape, result->rank);
    int elements_along_axis = tensor->shape[axis];
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    sum_axis_ptask tasks[num_threads];
    
    // Calculate optimal chunk sizes for result elements
    size_t chunk = result_numel / num_threads;
    size_t remainder = result_numel % num_threads;
    
    // Configure tasks
    for (size_t i = 0; i < num_threads; i++) {
        tasks[i].src_data = tensor->data;
        tasks[i].dst_data = result->data;
        tasks[i].tensor = tensor;
        tasks[i].result = result;
        tasks[i].axis = axis;
        tasks[i].elements_along_axis = elements_along_axis;
        tasks[i].result_numel = result_numel;
        tasks[i].dtype = tensor->dtype;
    }
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        
        int status = pthread_create(&threads[i], NULL, nnl2_own_paxis_sum, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sum_with_axis");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

void* nnl2_own_paxis_sum(void* arg) {
    sum_axis_ptask* task = (sum_axis_ptask*)arg;
    
    switch(task->dtype) {
        case FLOAT64: {
            double* data = (double*)task->src_data;
            double* result_data = (double*)task->dst_data;
            
            for (size_t i = task->start_idx; i < task->end_idx; i++) {
                double sum = 0.0;
                for (int k = 0; k < task->elements_along_axis; k++) {
                    size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(
                        task->tensor, task->result, task->axis, i, k);
                    sum += data[original_index];
                }
                result_data[i] = sum;
            }
			
            break;
        }
		
        case FLOAT32: {
            float* data = (float*)task->src_data;
            float* result_data = (float*)task->dst_data;
            
            for (size_t i = task->start_idx; i < task->end_idx; i++) {
                float sum = 0.0f;
                for (int k = 0; k < task->elements_along_axis; k++) {
                    size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(
                        task->tensor, task->result, task->axis, i, k);
                    sum += data[original_index];
                }
                result_data[i] = sum;
            }
			
            break;
        }
		
        case INT32: {
            int32_t* data = (int32_t*)task->src_data;
            int32_t* result_data = (int32_t*)task->dst_data;
            
            for (size_t i = task->start_idx; i < task->end_idx; i++) {
                int32_t sum = 0;
                for (int k = 0; k < task->elements_along_axis; k++) {
                    size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(
                        task->tensor, task->result, task->axis, i, k);
                    sum += data[original_index];
                }
                result_data[i] = sum;
            }
			
            break;
        }
		
        default: break;
    }
    
    return NULL;
}

#endif

/** @brief
 * See docs at declaration
 *
 ** @see naive_calculate_original_index_for_sum_with_axis
 **/
NNL2_FORCE_INLINE static size_t nnl2_naive_calculate_original_index_for_sum_with_axis(Tensor* tensor, Tensor* result, int axis, size_t result_index, int axis_index) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif

    size_t original_index = 0;
    int temp = (int)result_index;

    bool keepdim = (result->rank == tensor->rank);

    for (int dim = result->rank - 1; dim >= 0; dim--) {
        int coord = temp % result->shape[dim];
        temp /= result->shape[dim];

        int original_dim;
        if (keepdim) {
            original_dim = dim;
        } else {
            original_dim = (dim >= axis) ? (dim + 1) : dim;
        }

        original_index += (size_t)coord * tensor->strides[original_dim];
    }

    original_index += (size_t)axis_index * tensor->strides[axis];

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif

    return original_index;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float64
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float64(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    double* data = (double*)tensor->data;
    double* result_data = (double*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        double sum = 0.0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_float32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_float32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
    float* data = (float*)tensor->data;
    float* result_data = (float*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        float sum = 0.0f;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * See docs at declaration
 *
 ** @see naive_sum_with_axis_int32
 **/
NNL2_FORCE_INLINE static Tensor* nnl2_naive_sum_with_axis_int32(Tensor* tensor, Tensor* result, int axis, size_t result_numel, int elements_along_axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	int32_t* data = (int32_t*)tensor->data;
    int32_t* result_data = (int32_t*)result->data;
    
	// Iterate over all elements in the result tensor
    for (size_t i = 0; i < result_numel; i++) {
        int32_t sum = 0;
		
		// Sum along the specified axis
        for (int k = 0; k < elements_along_axis; k++) {
            size_t original_index = nnl2_naive_calculate_original_index_for_sum_with_axis(tensor, result, axis, i, k);
            sum += data[original_index];
        }
		
		// Store the computed sum
        result_data[i] = sum;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for sum with axis operation
 * @details
 * Array follows the common backend registration pattern for axis-wise
 * summation operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor summation along specified axes
 * 
 * @see nnl2_naive
 * @see naive_sum_with_axis
 */
Implementation sum_with_axis_backends[] = {
    REGISTER_BACKEND(naive_sum_with_axis, nnl2_naive, NAIVE_BACKEND_NAME),
    
    #ifdef NNL2_PTHREAD_AVAILABLE
        REGISTER_BACKEND(nnl2_own_sum_with_axis, nnl2_own, NNL2_OWN_NAME),
    #endif
};	

/**
 * @brief Function pointer type for sum with axis operation with keepdim
 */
typedef Tensor* (*sumwithaxiskeepdimfn)(Tensor*, int, bool);

/**
 * @brief Function pointer for sum with axis operation with keepdim
 * @ingroup backend_system 
 * 
 * This function pointer is used to call the currently active backend
 * implementation for summing tensor elements along specified dimensions.
 */
sumwithaxiskeepdimfn nnl2_sum_with_axis;

/** 
 * @brief Makes the sum with axis backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(sum_with_axis);

/** 
 * @brief Sets the backend for sum with axis operation
 * @ingroup backend_system
 * @param[in] backend_name Name of the backend to activate for axis-wise summation
 * @see ESET_BACKEND_BY_NAME
 * 
 * This function allows dynamic switching between different backend implementations
 * for the sum with axis operation at runtime.
 */
void set_sum_with_axis_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sum_with_axis_backends, nnl2_sum_with_axis, backend_name, CURRENT_BACKEND(sum_with_axis));
}

#endif /** NNL2_SUM_H **/
