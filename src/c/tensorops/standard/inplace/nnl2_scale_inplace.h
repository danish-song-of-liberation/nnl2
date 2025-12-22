#ifndef NNL2_SCALE_INPLACE_H
#define NNL2_SCALE_INPLACE_H

/** @brief
 * Scales the tensor in-place using a multiplier
 * The function multiplies each element of the tensor by a specified multiplier
 *
 ** @param tensor 	
 * Pointer to a tensor for scaling
 *
 ** @param multiplier
 * Multiplier for scaling
 *
 ** @note
 * For integer types, the multiplier must be an integer
 *
 ** @throws NNL2_ERROR
 * If you try to multiply an INT32 tensor by a fractional number
 */
void naive_scaleinplace(nnl2_tensor* tensor, float multiplier) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	void* data = tensor->data;
	
	// Calculate the total number of elements in the tensor
	int num_elems = nnl2_product(tensor->shape, tensor->rank);
	if(num_elems == 0) return; // If the tensor is empty, exit
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* data_t = (double*)data;
			for(int i = 0; i < num_elems; i++) data_t[i] *= (double)multiplier;
			break;
		}
		
		case FLOAT32: {
			float* data_t = (float*)data;
			for(int i = 0; i < num_elems; i++) data_t[i] *= multiplier;
			break;
		}
		
		case INT32: {
			int32_t* data_t = (int32_t*)data;
			
			// Check that the multiplier is integer for the INT32 tensor
			if(round(multiplier) != multiplier) {
				NNL2_ERROR("Can't multiply an int32 tensor by a fractional number");
				return;
			}
			
			for(int i = 0; i < num_elems; i++) data_t[i] = (int32_t)round(data_t[i] * multiplier);
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

/** 
 * @ingroup backend_system
 * @brief Backend implementations for scaleinplace operation
 * @details
 * Array follows the common backend registration pattern for scaleinplace operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
nnl2_runtime_implementation scaleinplace_backends[] = {
	REGISTER_BACKEND(naive_scaleinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for scaleinplace operation
 * @ingroup backend_system 
 */
scaleinplacefn scaleinplace;

/** 
 * @brief Makes the scaleinplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(scaleinplace);

/** 
 * @brief Sets the backend for scaleinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see ESET_BACKEND_BY_NAME
 */
void set_scaleinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(scaleinplace_backends, scaleinplace, backend_name, CURRENT_BACKEND(scaleinplace));
}

/** 
 * @brief Gets the name of the active backend for scaleinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_scaleinplace_backend() {
	return CURRENT_BACKEND(scaleinplace);
}

/** 
 * @brief Function declaration for getting all available scaleinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(scaleinplace);

/**
 * @brief Function declaration for getting the number of available scaleinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(scaleinplace);

#endif /** NNL2_SCALE_INPLACE_H **/
