#ifndef NNL2_KAIMING_INPLACE_H
#define NNL2_KAIMING_INPLACE_H

#define NNL2_KAIMING_MODE_FAN_IN 0
#define NNL2_KAIMING_MODE_FAN_OUT 1
#define NNL2_KAIMING_MODE_FAN_AVG 2

/** @brief
 * In-place Kaiming (He) initialization of a tensor
 *
 ** @param tensor
 * Pointer to the tensor to initialize
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
 ** @see product
 ** @see sqrt
 **/
void nnl2_naive_kaiming_inplace(nnl2_tensor* tensor, int fan_in, int fan_out, float gain, float distribution, int mode) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In nnl2_naive_kaiming_inplace, tensor is NULL");
    #endif

    if(tensor->dtype == INT32) {
        NNL2_FATAL("INT32 can't be used for Kaiming initialization");
        return;
    }

    size_t total_elems = product(tensor->shape, tensor->rank);
    if(total_elems == 0) return;

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
        return;
    }
    
    float scale_param = gain * sqrt(distribution / denominator);
	
	if(fabsf(distribution - 6.0f) < 1e-6f) {
		double from = (double)-scale_param;
		double to = (double)scale_param;
		
        uniform_inplace(tensor, &from, &to);
		
    } else {
        randn_inplace(tensor, 0.0, (double)scale_param);
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for in-place Kaiming operation
 * @details
 * Array follows the common backend registration pattern for Kaiming initialization
 * operations. Currently registered backends:
 *  - nnl2_naive_kaiming_inplace: Basic reference implementation for in-place Kaiming initialization
 *
 * @see nnl2_naive
 * @see nnl2_naive_kaiming_inplace
 */
Implementation kaiming_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_kaiming_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for in-place Kaiming operation
 * @ingroup backend_system
 */
kaiminginplacefn kaiming_inplace;

/**
 * @brief Makes the in-place Kaiming backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(kaiming_inplace);

/**
 * @brief Sets the backend for in-place Kaiming operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for in-place Kaiming
 * @see ESET_BACKEND_BY_NAME
 */
void set_kaiming_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(kaiming_inplace_backends, kaiming_inplace, backend_name, CURRENT_BACKEND(kaiming_inplace));
}

/**
 * @brief Gets the name of the active backend for in-place Kaiming operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_kaiming_inplace_backend() {
    return CURRENT_BACKEND(kaiming_inplace);
}

/**
 * @brief Function declaration for getting all available in-place Kaiming backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(kaiming_inplace);

/**
 * @brief Function declaration for getting the number of available in-place Kaiming backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(kaiming_inplace);

#endif /** NNL2_KAIMING_INPLACE_H **/
