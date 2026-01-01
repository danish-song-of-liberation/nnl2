#ifndef NNL2_SOFTMAX_H
#define NNL2_SOFTMAX_H

/** @brief 
 * Computes the softmax function along a specified dimension.
 * 
 ** @param input 
 * Input tensor
 *
 ** @param dim 
 * Dimension along which softmax will be computed
 * Must be in the range [0, input->rank - 1]
 * 
 ** @return nnl2_tensor*
 * A new tensor with the same shape as input containing softmax values
 */
nnl2_tensor* nnl2_naive_softmax(nnl2_tensor* input, int dim) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if(input == NULL) {
			NNL2_ERROR("In function nnl2_naive_softmax, passed tensor is NULL. returning NULL");
			return NULL;
		}
		
		if (dim < 0 || dim >= input->rank) {
			NNL2_ERROR("In function nnl2_naive_softmax, invalid dimension %d for tensor of rank %d", dim, input->rank);
			return NULL;
		}
	#endif 
	
	nnl2_tensor* output = nnl2_empty_like(input);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if(output == NULL) {
			NNL2_ERROR("In function nnl2_naive_softmax, failed to allocate output tensor. returning NULL");
			return NULL;
		}
	#endif 
	
	int32_t dim_size = input -> shape[dim];
	int32_t stride_dim = input -> strides[dim];
	
	switch(input->dtype) {
        case FLOAT32: {
			nnl2_float32* input_data = (nnl2_float32*)input->data;
			nnl2_float32* output_data = (nnl2_float32*)output->data;
			size_t total_elements = nnl2_product(input->shape, input->rank);
			
			for(size_t idx = 0; idx < total_elements; idx += dim_size * stride_dim) {
				// Find maximum value in the current softmax group
				nnl2_float32 max_val = input_data[idx];
				for(int32_t k = 1; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					if(input_data[k_idx] > max_val) {
						max_val = input_data[k_idx];
					}
				}
				
				nnl2_float32 exp_sum = 0.0f;
				for(int32_t k = 0; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					nnl2_float32 exp_val = expf(input_data[k_idx] - max_val);
					output_data[k_idx] = exp_val;  
					exp_sum += exp_val;
				}
				
				for(int32_t k = 0; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					output_data[k_idx] /= exp_sum;
				}
			}
			
			break;
		}
		
		case FLOAT64: {
			nnl2_float64* input_data = (nnl2_float64*)input->data;
			nnl2_float64* output_data = (nnl2_float64*)output->data;
			size_t total_elements = nnl2_product(input->shape, input->rank);
			
			for(size_t idx = 0; idx < total_elements; idx += dim_size * stride_dim) {
				nnl2_float32 max_val = input_data[idx];
				for(int32_t k = 1; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					if(input_data[k_idx] > max_val) {
						max_val = input_data[k_idx];
					}
				}
				
				nnl2_float64 exp_sum = 0.0f;
				for(int32_t k = 0; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					nnl2_float64 exp_val = exp(input_data[k_idx] - max_val);
					output_data[k_idx] = exp_val;  
					exp_sum += exp_val;
				}
				
				for(int32_t k = 0; k < dim_size; k++) {
					int32_t k_idx = idx + k * stride_dim;
					output_data[k_idx] /= exp_sum;
				}
			}
			
			break;
		}
		
		default: {
            NNL2_TYPE_ERROR(input->dtype);
            nnl2_free_tensor(output);
            return NULL;
        }
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif 
	
	return output;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for softmax operation
 * @details
 * Array follows the standard backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_softmax: Basic reference implementation
 *
 * @see nnl2_naive_softmax
 */
nnl2_runtime_implementation softmax_backends[] = {
    REGISTER_BACKEND(nnl2_naive_softmax, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer type for softmax operation
 * @ingroup backend_system
 */
typedef nnl2_tensor* (*softmaxfn)(nnl2_tensor* input, int dim);

/**
 * @brief Function pointer for softmax operation
 * @ingroup backend_system
 */
softmaxfn nnl2_softmax;

/**
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(softmax);

/**
 * @brief Sets the backend for softmax operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_softmax_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(softmax_backends, nnl2_softmax, backend_name, CURRENT_BACKEND(softmax));
}

/**
 * @brief Gets the name of the active backend for softmax operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_softmax_backend() {
    return CURRENT_BACKEND(softmax);
}

/**
 * @brief Function declaration for getting all `softmax` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(softmax);

/**
 * @brief Function declaration for getting the number of all `softmax` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(softmax);

#endif /** NNL2_SOFTMAX_H **/
