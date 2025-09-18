#ifndef NNL2_LEAKY_RELY_H
#define NNL2_LEAKY_RELY_H

/** @brief
 * A function that does what you already know with a tensor, but documentation is required
 *
 ** @details
 * Applies Leaky ReLU (max(x * alpha, x)) function element-wise
 *
 * Example 1: leakyrelu(1) -> 1
 * Example 2: leakyrelu(-1, alpha=0.01) -> -0.01
 *
 ** @param tensor
 * Input tensor to apply Leaky ReLU to
 *
 ** @param alpha
 * Negative slope coefficient for values less than zero
 *
 ** @param save_type
 * Tries to preserve the initial tensor type if possible. It is recommended to set true
 *
 ** @return
 * New tensor with Leaky ReLU applied
 */
Tensor* naive_leakyrelu(Tensor* tensor, float alpha, bool save_type) {	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Calculate total number of elements
	int total_elems = product(tensor->shape, tensor->rank);	
	Tensor* result = nnl2_empty(tensor->shape, tensor->rank, tensor->dtype); 
	
	if(total_elems == 0) return result; // If tensor is empty then return new empty tensor
	
	void* data_t = tensor->data;
	void* data_r = result->data;
	
	bool float64_conversion = false;
	
	switch(tensor->dtype) {
		case FLOAT64: {
			double* cast_data_t = (double*)data_t;	
			double* cast_data_r = (double*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float64(cast_data_t[i], alpha);
			break;
		}
		
		case FLOAT32: {
			float* cast_data_t = (float*)data_t;	
			float* cast_data_r = (float*)data_r;
			for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_float32(cast_data_t[i], alpha);
			break;
		}
		
		case INT32: {
			int32_t* cast_data_t = (int32_t*)data_t;	
			int32_t* cast_data_r = (int32_t*)data_r;
			
			// Check if conversion to float64 is needed for int32
			for(int i = 0; i < total_elems; i++) {
				if(cast_data_t[i] < 0) {
					float result_val = cast_data_t[i] * alpha;
					
					// Check if result has fractional part
					if(fmodf(result_val, 1.0f) != 0.0f) {
						float64_conversion = true;
						break;
					}
				}
			}
			
			// Handle int32 with potential type conversion
			if(float64_conversion || !save_type) {
				nnl2_free_tensor(result); 
				
				result = nnl2_empty(tensor->shape, tensor->rank, FLOAT64); // this is, by the way, ineffective, but I'm too lazy to redo it
				
				// I don't care if you need performance, I'll only add it if I need it myself
				
				// Besides, in that case, I'll probably abandon this implementation 
				// and do the parallelization right away
				
				data_r = result->data;
				
				double* cast_data_r_f64 = (double*)data_r;
				
				for(int i = 0; i < total_elems; i++) {
					if(cast_data_t[i] >= 0) {
						cast_data_r_f64[i] = (double)cast_data_t[i];
					} else {
						cast_data_r_f64[i] = (double)(cast_data_t[i] * alpha);
					}
				}
			} else {
				for(int i = 0; i < total_elems; i++) cast_data_r[i] = nnl2_leaky_relu_int32(cast_data_t[i], alpha); // oh my god, there are extra int checks in every call, it's so unnecessary
			}
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return NULL;	
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for LeakyReLU operation
 * @details
 * Array follows the common backend registration pattern for LeakyReLU operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for LeakyReLU activation function
 * 
 * @see nnl2_naive
 * @see naive_leakyrelu
 */
Implementation leakyrelu_backends[] = {
	REGISTER_BACKEND(naive_leakyrelu, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for LeakyReLU operation
 * @ingroup backend_system 
 */
leakyrelufn leakyrelu;

/** 
 * @brief Makes the leakyrelu backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(leakyrelu);

/** 
 * @brief Sets the backend for LeakyReLU operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for LeakyReLU
 * @see ESET_BACKEND_BY_NAME
 */
void set_leakyrelu_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(leakyrelu_backends, leakyrelu, backend_name, current_backend(leakyrelu));
}

/** 
 * @brief Gets the name of the active backend for LeakyReLU operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_leakyrelu_backend() {
	return current_backend(leakyrelu);
}

/** 
 * @brief Function declaration for getting all available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(leakyrelu);

/**
 * @brief Function declaration for getting the number of available LeakyReLU backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(leakyrelu);

#endif /** NNL2_LEAKY_RELY_H **/
