#ifndef NNL2_SIGMOID_INPLACE_H
#define NNL2_SIGMOID_INPLACE_H

///@{
	
/** @brief
 * Threshold for enabling parallel execution of the
 * Sigmoid operation during in-place calculations
 */
#define NNL2_SIGMOID_INPLACE_PARALLEL_THREASHOLD 1000000

///@}

/** @brief
 * Applies a sigmoid function to a tensor in place
 *
 ** @details
 * Sigmoid function: sigmoid(x) = 1 / (1 + exp(-x))
 *
 * Example 1: sigmoid(1) = 1 / (1 + exp(-1)) ≈ 0.731
 * Example 2: sigmoid(0.5) = 1 / (1 + exp(-0.5)) ≈ 0.622
 *
 ** @param tensor
 * A pointer to a tensor for conversion
 *
 ** @see nnl2_product
 **/
void naive_sigmoidinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	size_t total_elems = product(tensor->shape, tensor->rank);	
	if(total_elems == 0) return; // If 0 elems then return
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data = (double*)data;	
			for(size_t i = 0; i < total_elems; i++) nnl2_sigmoid_float64_inplace(&cast_data[i]);
			break;
		}
		
		case FLOAT32: {
			float* cast_data = (float*)data;	
			for(size_t i = 0; i < total_elems; i++) nnl2_sigmoid_float32_inplace(&cast_data[i]);
			break;
		}
		
		case INT32: {
			NNL2_FATAL("Sigmoid in-place cannot be applied to the provided tensor");
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
 * Multithreaded in-place Sigmoid for double (float64) precision floating-point arrays
 * 
 * @param data 
 * Pointer to double array
 *
 * @param total_size 
 * Total number of elements in the array
 *
 * @param nthreads 
 * Number of threads for parallelization
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_sigmoid_inplace_float64(double* data, size_t total_size, size_t nthreads);

/** @brief 
 * Similarly nnl2_own_sigmoid_inplace_float64 but for float32
 *
 ** @see nnl2_own_sigmoid_inplace_float64
 **/
void* nnl2_own_sigmoid_inplace_float32(float* data, size_t total_size,  size_t nthreads);

/** @brief 
 * Worker function wrapper for parallel Sigmoid execution on float64 arrays
 * 
 * @param arg 
 * Pointer to single_arr_ptask structure containing task parameters
 *
 * @return NULL (for pthread api)
 */
void* nnl2_own_psigmoid_inplace_float64(void* arg);

/** @brief 
 * Similarly nnl2_own_psigmoid_inplace_float64 but for float32
 *
 ** @see nnl2_own_psigmoid_inplace_float64
 **/
void* nnl2_own_psigmoid_inplace_float32(void* arg);

/** @brief
 * Main function for applying Sigmoid activation to tensor in-place
 * 
 ** @param tensor 
 * Pointer to tensor to be modified
 */
void nnl2_own_sigmoid_inplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);	
	
	if(total_elems == 0) {
		return; // If tensor is empty return
	} else if (total_elems < NNL2_SIGMOID_INPLACE_PARALLEL_THREASHOLD) {
		naive_sigmoidinplace(tensor);
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_FUNC_EXIT();
		#endif
		return;
	}
	
	void* data = tensor->data;
	
	switch(tensor->dtype) {
		case FLOAT64: nnl2_own_sigmoid_inplace_float64((double*)data, total_elems, NNL2_NUM_THREADS);  break;
		case FLOAT32: nnl2_own_sigmoid_inplace_float32((float*)data, total_elems,  NNL2_NUM_THREADS);  break;
		case   INT32: {
			NNL2_FATAL("Sigmoid in-place cannot be applied to the provided tensor");
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

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_sigmoid_inplace_float64
 **/
void* nnl2_own_sigmoid_inplace_float64(double* data, size_t total_size, size_t num_threads) {
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    single_arr_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, nnl2_own_psigmoid_inplace_float64, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid_inplace_float64");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sigmoid_inplace_float64");
        }
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_sigmoid_inplace_float32
 **/
void* nnl2_own_sigmoid_inplace_float32(float* data, size_t total_size, size_t num_threads) {
    // Allocate arrays for thread handles and task descriptors
    pthread_t threads[num_threads];
    single_arr_ptask tasks[num_threads];
    
    // Calculate base chunk size and remainder for balanced distribution
    size_t chunk = total_size / num_threads;
    size_t remainder = total_size % num_threads;
    
    // Distribute work among threads with load balancing
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        // Configure task for this thread
        tasks[i].data = data;
        tasks[i].start = current_start;
        tasks[i].end = current_start + current_chunk;
        
        // Create thread to process the assigned chunk
        int status = pthread_create(&threads[i], NULL, nnl2_own_psigmoid_inplace_float32, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_sigmoid_inplace_float32");
            num_threads = i;
            break;
        }
        
        current_start += current_chunk;
    }
    
    // Wait for all threads to complete their work
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_sigmoid_inplace_float32");
        }
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_psigmoid_inplace_float64
 **/
void* nnl2_own_psigmoid_inplace_float64(void* arg) {
    // Extract task parameters from argument
    single_arr_ptask* task = (single_arr_ptask*)arg;
    double* input = (double*)task->data;
    
    // Apply Sigmoid activation to each element in the assigned range
    for (size_t i = task->start; i < task->end; i++) {
        nnl2_sigmoid_float64_inplace(&input[i]);
    }
    
    return NULL;
}

/** @brief
 * See docs at declaration
 *
 ** @see nnl2_own_psigmoid_inplace_float32
 **/
void* nnl2_own_psigmoid_inplace_float32(void* arg) {
    // Extract task parameters from argument
    single_arr_ptask* task = (single_arr_ptask*)arg;
    float* input = (float*)task->data;
    
    // Apply Sigmoid activation to each element in the assigned range
    for (size_t i = task->start; i < task->end; i++) {
        nnl2_sigmoid_float32_inplace(&input[i]);
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for sigmoidinplace operation
 * @details
 * Array follows the common backend registration pattern for sigmoidinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place sigmoid activation function
 *  - nnl2_own: Own nnl2 implementation for in-place sigmoid activation
 * 
 * @see nnl2_naive
 * @see naive_sigmoidinplace
 */
Implementation sigmoidinplace_backends[] = {
	REGISTER_BACKEND(naive_sigmoidinplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_sigmoid_inplace, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for sigmoidinplace operation
 * @ingroup backend_system 
 */
sigmoidinplacefn sigmoidinplace;

/** 
 * @brief Makes the sigmoidinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(sigmoidinplace);

/** 
 * @brief Sets the backend for sigmoidinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for sigmoidinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_sigmoidinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(sigmoidinplace_backends, sigmoidinplace, backend_name, current_backend(sigmoidinplace));
}

/** 
 * @brief Gets the name of the active backend for sigmoidinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_sigmoidinplace_backend() {
	return current_backend(sigmoidinplace);
}

/** 
 * @brief Function declaration for getting all available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(sigmoidinplace);

/**
 * @brief Function declaration for getting the number of available sigmoidinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(sigmoidinplace);

#endif /** NNL2_SIGMOID_INPLACE_H **/
