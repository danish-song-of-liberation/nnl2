#ifndef NNL2_HSTACK_H
#define NNL2_HSTACK_H

/** @brief 
 * Performs horizontal stacking of two tensors (naive implementation)
 *
 ** @param tensora
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @return
 * Pointer to a new tensor containing the horizontally stacked result
 *
 ** @note
 * Tensors must have the same rank and compatible shapes (all dimensions
 * except axis=1 must match)
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_hstack(Tensor* tensora, Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;

	// Safety checks for tensor compatibility
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(ranka != rankb) {
			NNL2_ERROR("Tensors dimensions are different");
			return NULL;
		}

		// Check if all dimensions except axis=1 are equal
		for(int i = 0; i < ranka; i++) {
			if(i != 1 && tensora->shape[i] != tensorb->shape[i]) {
				NNL2_ERROR("Tensors shapes are incompatible for hstack");
				return NULL;
			}
		}
	#endif

    TensorType winner_type = MAX(typea, typeb);

	// Allocate memory for result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed");
        return NULL; 
    }
    
	// Calculate result shape
    for(int i = 0; i < ranka; i++) {
        if(i == 1) {
            shapec[i] = tensora->shape[i] + tensorb->shape[i];
        } else {
            shapec[i] = tensora->shape[i];
        }
    }
    
	// Create empty result tensor with calculated shape and winning type
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor in hstack");
        return NULL;
    }
    
	// Handle empty tensors case
    size_t total_elements = product(result->shape, result->rank);
    if(total_elements == 0) {
        return result;
    }
    
    size_t shapea_0 = (size_t)tensora->shape[0];
    size_t shapeb_0 = (size_t)tensorb->shape[0];
    
	// Handle 1D tensors
    if(ranka == 1) {
        if(typea == typeb && typea == winner_type) {
			// Same types, use memcpy
            size_t item_size = get_dtype_size(winner_type);
            memcpy(result->data, tensora->data, shapea_0 * item_size);
            memcpy((char*)result->data + shapea_0 * item_size, tensorb->data, shapeb_0 * item_size);
        } else {
		    // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;

					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)tensora->data + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)tensorb->data + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    free(result);
                    return NULL;
                }
            }
        }
    } else {
		// Handle multi-dimensional tensors
        size_t outer_dim = (size_t)tensora->shape[0];
        size_t inner_elements_a = product(tensora->shape + 1, ranka - 1);
        size_t inner_elements_b = product(tensorb->shape + 1, rankb - 1);
        
        if(typea == typeb && typea == winner_type) {
			// Same types, use memcpy 
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size_a = inner_elements_a * item_size;
            size_t row_size_b = inner_elements_b * item_size;
            
            char* src_a = tensora->data;
            char* src_b = tensorb->data;
            char* dst = result->data;
            
			// Process each outer dimension (e.g., each matrix in a 3D tensor)
            for(size_t i = 0; i < outer_dim; i++) {
				// Copy slice from first tensor
                memcpy(dst, src_a, row_size_a);
                dst += row_size_a;
                src_a += row_size_a;
                
				// Copy slice from second tensor
                memcpy(dst, src_b, row_size_b);
                dst += row_size_b;
                src_b += row_size_b;
            }
        } else {
			// Type conversion needed for multi-dimensional case
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typea);
                        }
						
                        // Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typea);
                        }
                        
						// Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Process each outer dimension
                    for(size_t i = 0; i < outer_dim; i++) {
						// Convert and copy slice from first tensor
                        for(size_t j = 0; j < inner_elements_a; j++) {
                            size_t src_idx = i * inner_elements_a + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + j;
                            
                            void* elem = (char*)tensora->data + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typea);
                        }
                        
						// Convert and copy slice from second tensor
                        for(size_t j = 0; j < inner_elements_b; j++) {
                            size_t src_idx = i * inner_elements_b + j;
                            size_t dst_idx = i * (inner_elements_a + inner_elements_b) + inner_elements_a + j;
                            
                            void* elem = (char*)tensorb->data + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
					
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    free(result);
                    return NULL;
                }
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for hstack operation
 * @details
 * Array follows the common backend registration pattern for horizontal stacking operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for horizontal tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_hstack
 */
Implementation hstack_backends[] = {
	REGISTER_BACKEND(naive_hstack, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for hstack operation
 * @ingroup backend_system 
 */
hstackfn hstack;

/** 
 * @brief Makes the hstack backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(hstack);

/** 
 * @brief Sets the backend for hstack operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for horizontal stacking
 * @see ESET_BACKEND_BY_NAME
 */
void set_hstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(hstack_backends, hstack, backend_name, current_backend(hstack));
}

/** 
 * @brief Gets the name of the active backend for hstack operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_hstack_backend() {
	return current_backend(hstack);
}

/** 
 * @brief Function declaration for getting all available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(hstack);

/**
 * @brief Function declaration for getting the number of available hstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 * @details
 * Returns the total number of registered horizontal stacking backend implementations.
 * Useful for iterating through available backends.
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(hstack);

#endif /** NNL2_HSTACK_H **/
