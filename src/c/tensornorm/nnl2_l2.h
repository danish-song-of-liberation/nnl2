#ifndef NNL2_L2_H
#define NNL2_L2_H

/** @brief
 * Computes the L2 norm (Euclidean norm) of a tensor
 *
 ** @param tensor
 * Pointer to the input tensor
 *
 ** @param axes
 * Array of axes for reduction
 *
 ** @param num_axes
 * Number of axes for reduction
 *
 ** @param result
 * Pointer to memory for storing the result
 *
 ** @see sqrt
 ** @see sqrtf
 **/
void naive_l2norm(Tensor* tensor, int* axes, int num_axes, void* result) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	bool reduce_all = false;
	
	if(num_axes == 1 && axes[0] == 0) {
		reduce_all = true;
	} 
	
	if(reduce_all) {
		size_t total_elems = product(tensor->shape, tensor->rank);
		
		switch(tensor->dtype) {
			case FLOAT64: {
                double* cast_data = (double*)tensor->data;
                double acc = 0.0;
                for (size_t it = 0; it < total_elems; it++) {
                    double val = cast_data[it];
                    acc += val * val;
                }
                *((double*)result) = sqrt(acc); // Store result in output pointer
                break;
            }
    
            case FLOAT32: {
                float* cast_data = (float*)tensor->data;
                float acc = 0.0f;
                for (size_t it = 0; it < total_elems; it++) {
                    float val = cast_data[it];
                    acc += val * val;
                }
                *((float*)result) = sqrtf(acc); 
                break;
            }
            
            case INT32: {
                int32_t* cast_data = (int32_t*)tensor->data;
                int32_t acc = 0;
                for (size_t it = 0; it < total_elems; it++) {
                    int32_t val = cast_data[it];
                    acc += val * val; 
                }
                *((int32_t*)result) = (int32_t)sqrt(acc);  
                break;
            }
			
			default: {
				NNL2_TYPE_ERROR(tensor->dtype);
				return;
			}
		}
	} else {
		NNL2_ERROR("Norm axes in development");
		
		// I don't plan to add support for axes 
		// in the near future because I don't need it
		
		return;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for L2 normalization operation
 * @details
 * Array follows the common backend registration pattern for L2 normalization
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for L2 normalization
 * 
 * @see nnl2_naive
 * @see naive_l2norm
 */
Implementation l2norm_backends[] = {
	REGISTER_BACKEND(naive_l2norm, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for L2 normalization operation
 * @ingroup backend_system 
 */
l2normfn l2norm;

/** 
 * @brief Makes the L2 normalization backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
MAKE_CURRENT_BACKEND(l2norm);

/** 
 * @brief Sets the backend for L2 normalization operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for L2 normalization
 * @see ESET_BACKEND_BY_NAME
 */
void set_l2norm_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(l2norm_backends, l2norm, backend_name, CURRENT_BACKEND(l2norm));
}

/** 
 * @brief Gets the name of the active backend for L2 normalization operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_l2norm_backend() {
	return CURRENT_BACKEND(l2norm);
}

/** 
 * @brief Function declaration for getting all available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(l2norm);

/**
 * @brief Function declaration for getting the number of available L2 normalization backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(l2norm);

#endif /** NNL2_L2_H **/
