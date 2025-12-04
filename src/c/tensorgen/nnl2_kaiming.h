#ifndef NNL2_KAIMING_H
#define NNL2_KAIMING_H

#define NNL2_KAIMING_MODE_FAN_IN 0
#define NNL2_KAIMING_MODE_FAN_OUT 1
#define NNL2_KAIMING_MODE_FAN_AVG 2

#define NNL2_KAIMING_NO_GAIN 1.0f

#define NNL2_KAIMING_NORMAL_DIST 2.0f
#define NNL2_KAIMING_UNIFORM_DIST 6.0f

/** @brief
 * Initializing a tensor using the Kaiming (He) distribution
 *
 ** @param shape
 * Pointer to an array of tensor dimensions
 *
 ** @param rank
 * The rank of the tensor
 *
 ** @param dtype
 * Tensor data type
 *
 ** @param fan_in
 * Number of input neurons
 *
 ** @param fan_out
 * Number of output neurons
 *
 ** @param gain
 * Gain factor (usually sqrt(2.0) for ReLU)
 *
 ** @param distribution
 * Distribution parameter: 
 *  2.0 = Normal distribution (Kaiming Normal)
 *  6.0 = Uniform distribution (Kaiming Uniform)
 *
 ** @param mode
 * Mode of initialization: 
 *  0 = "fan_in" (default), 
 *  1 = "fan_out",
 *  2 = "fan_avg" (average of fan_in and fan_out)
 *
 ** @note
 * Integer data types are not supported
 *
 ** @see RAND_MAX
 ** @see nnl2_empty
 **/
nnl2_tensor* nnl2_naive_kaiming(int* shape, int rank, nnl2_tensor_type dtype, int fan_in, int fan_out, float gain, float distribution, int mode) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(dtype == INT32) {
        NNL2_FATAL("INT32 Can't be used for kaiming distribution");
        return NULL;
    }
    
    nnl2_tensor* result = nnl2_empty(shape, rank, dtype);
    
    size_t total_elems = product(shape, rank);
    if(total_elems == 0) return result; // If tensor is empty return empty result

    // Calculate denominator based on mode
    float denominator;
    switch(mode) {
        case NNL2_KAIMING_MODE_FAN_IN:     denominator = (float)fan_in;   				     break;
        case NNL2_KAIMING_MODE_FAN_OUT:    denominator = (float)fan_out;   					 break;
        case NNL2_KAIMING_MODE_FAN_AVG:    denominator = (float)(fan_in + fan_out) / 2.0f;   break;
		
        default: {
            NNL2_WARN("Unknown mode %d, using fan_in (mode=0)", mode);
            denominator = (float)fan_in;
		}
    }
    
    if(denominator <= 0.0f) {
        NNL2_FATAL("Denominator must be positive, got %f", denominator);
        nnl2_free_tensor(result);
        return NULL;
    }
    
    float scale_param = gain * sqrtf(distribution / denominator);
	
	if(fabsf(distribution - 6.0f) < 1e-6f) {
		double from = (double)-scale_param;
		double to = (double)scale_param;
		
        uniform_inplace(result, &from, &to);
		
    } else {
        randn_inplace(result, 0.0, (double)scale_param);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for kaiming operation
 * @details
 * Array follows the common backend registration pattern for kaiming initialization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for kaiming initialization
 * 
 * @see nnl2_naive
 * @see nnl2_naive_kaiming
 */
Implementation kaiming_backends[] = {
    REGISTER_BACKEND(nnl2_naive_kaiming, nnl2_naive, NAIVE_BACKEND_NAME),
};    

/**
 * @brief Function pointer for kaiming operation
 * @ingroup backend_system 
 */
kaimingfn kaiming;

/** 
 * @brief Makes the kaiming backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(kaiming);

/** 
 * @brief Sets the backend for kaiming operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for kaiming
 * @see ESET_BACKEND_BY_NAME
 */
void set_kaiming_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(kaiming_backends, kaiming, backend_name, current_backend(kaiming));
}

/** 
 * @brief Gets the name of the active backend for kaiming operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_kaiming_backend() {
    return current_backend(kaiming);
}

/** 
 * @brief Function declaration for getting all available kaiming backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(kaiming);

/**
 * @brief Function declaration for getting the number of available kaiming backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(kaiming);

#endif /** NNL2_KAIMING_H **/
