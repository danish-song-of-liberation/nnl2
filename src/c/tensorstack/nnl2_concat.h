#ifndef NNL2_CONCAT_H
#define NNL2_CONCAT_H

/** @brief
 * Performs concatenation of two tensors along specified axis (naive implementation)
 * Supports different data types with automatic type promotion
 *
 ** @details 
 * This function performs concatenation of two tensors along the given axis
 * The function supports automatic type promotion - if input tensors have
 * different data types, the result will use the "winner" type (higher precision)
 *
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @param axis
 * Axis along which to concatenate (0-based)
 *
 ** @return 
 * Pointer to a new tensor containing the concatenated result
 *
 ** @example
 * Tensor* a = ones(shape_a, rank, FLOAT32);
 * Tensor* b = ones(shape_b, rank, FLOAT32);
 * Tensor* result = naive_concat(a, b, 1); // Concatenate along axis 1
 * if (result != NULL) {
 *     // Use concatenated result
 *     nnl2_free_tensor(result);
 * }
 *
 * @retval NULL If ranks of input tensors does not match
 * @retval NULL If axis is out of valid range
 * @retval NULL If non-concatenation dimensions have incompatible shapes
 * @retval NULL If memory allocation fails
 * @retval NULL If unsupported data type encountered
 *
 ** @warning
 * DRY final boss 
 *
 ** @see get_dtype_size()
 ** @see nnl2_empty
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 ** @see nnl2_free_tensor()
 **/
Tensor* naive_concat(Tensor* tensora, Tensor* tensorb, int axis) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    
    TensorType winner_type = MAX(typea, typeb);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (ranka != rankb) {
            NNL2_ERROR("Ranks are different (%d != %d) in concat", ranka, rankb);
            return NULL;
        }

        if (axis < 0 || axis >= ranka) {
            NNL2_ERROR("Invalid axis %d (must be 0-%d) in concat", axis, ranka - 1);
            return NULL;
        }

        // Check compatible shapes using strides for efficiency
        for (int i = 0; i < ranka; i++) {
            if (i != axis && tensora->shape[i] != tensorb->shape[i]) {
                NNL2_ERROR("Incompatible shapes along axis %d (%d != %d) in concat", i, tensora->shape[i], tensorb->shape[i]);
                return NULL;
            }
        }
    #endif
    
    // Calculate result shape
    int* shapec = malloc(ranka * sizeof(int));
    if (shapec == NULL) {
        NNL2_ERROR("Memory allocation failed for shape in concat");
        return NULL;
    }
    
    for (int i = 0; i < ranka; i++) {
        shapec[i] = (i == axis) ? (tensora->shape[i] + tensorb->shape[i]) : tensora->shape[i];
    }    
    
    Tensor* result = nnl2_empty(shapec, ranka, winner_type);
    free(shapec);
    
    if (result == NULL) {
        return NULL;
    }
    
    size_t item_size = get_dtype_size(winner_type);
    
    // Use precomputed strides for efficient memory access
    size_t outer_elements = 1;
    for (int i = 0; i < axis; i++) {
        outer_elements *= tensora->shape[i];
    }
    
    size_t inner_elements = 1;
    for (int i = axis + 1; i < ranka; i++) {
        inner_elements *= tensora->shape[i];
    }
    
    size_t a_axis_elements = tensora->shape[axis];
    size_t b_axis_elements = tensorb->shape[axis];
    
    char* a_data = (char*)tensora->data;
    char* b_data = (char*)tensorb->data;
    char* c_data = (char*)result->data;
    
    if (typea == typeb && typea == winner_type) {
        // Handling case with same types, use memcpy with strides
        size_t a_slice_size = a_axis_elements * inner_elements * item_size;
        size_t b_slice_size = b_axis_elements * inner_elements * item_size;

        for (size_t outer = 0; outer < outer_elements; outer++) {
            size_t a_offset = outer * tensora->strides[axis] * a_axis_elements;
            size_t b_offset = outer * tensorb->strides[axis] * b_axis_elements;
            size_t c_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements);
            
            memcpy(c_data + c_offset, a_data + a_offset, a_slice_size);
            memcpy(c_data + c_offset + a_slice_size, b_data + b_offset, b_slice_size);
        }
    } else {
		// Now we handle the case when the variable names 
		// indicate that the code was written with AI support
		
        // Type conversion needed - process using strides
        switch(winner_type) { 
            case FLOAT64: {
				// Cast result data pointer 
                double* dst = (double*)result->data;
                
                for (size_t outer = 0; outer < outer_elements; outer++) {
					// Copy elements from first tensor (tensora)
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
						// Process all elements after the concatenation axis
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            // Calculate source offset in tensor A
							
							// outer * tensora->strides[axis] * a_axis_elements --- offset for outer dimensions
							// a_pos * tensora->strides[axis] --- offset along concatenation axis
							// inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1) --- offset for inner dimensions
					
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);
							
							// Calculate destination offset in result tensor
							// Similar calculation but with total axis size (a_axis_elements + b_axis_elements)
							
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
							// Get pointer to source element in tensor A
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to double and store in result
                            dst[dst_offset] = nnl2_convert_to_float64(elem, typea);
                        }
                    }
                    
					// Copy elements from second tensor (tensorb)
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            // Calculate source offset in tensor B
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
							
							// Calculate destination offset in result tensor
							// Position along concatenation axis is offset by a_axis_elements
							
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
							// Get pointer to source element in tensor B
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to double and store in result
                            dst[dst_offset] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            case FLOAT32: {
				// Cast result data pointer
                float* dst = (float*)result->data;
                
			    // Same nested loop structure as FLOAT64 case
                for (size_t outer = 0; outer < outer_elements; outer++) {
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to float32 instead of float64
                            dst[dst_offset] = nnl2_convert_to_float32(elem, typea);
                        }
                    }
                    
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to float32
                            dst[dst_offset] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            case INT32: {
				// Cast result data pointer
                int32_t* dst = (int32_t*)result->data;
                
			    // Same nested loop structure for integer concatenation
                for (size_t outer = 0; outer < outer_elements; outer++) {
                    for (size_t a_pos = 0; a_pos < a_axis_elements; a_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t a_src_offset = outer * tensora->strides[axis] * a_axis_elements + a_pos * tensora->strides[axis] + inner * (axis < ranka - 1 ? tensora->strides[axis + 1] : 1);                            
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + a_pos * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = a_data + a_src_offset * get_dtype_size(typea);
							
							// Convert element to int32
                            dst[dst_offset] = nnl2_convert_to_int32(elem, typea);
                        }
                    }
                    
                    for (size_t b_pos = 0; b_pos < b_axis_elements; b_pos++) {
                        for (size_t inner = 0; inner < inner_elements; inner++) {
                            size_t b_src_offset = outer * tensorb->strides[axis] * b_axis_elements + b_pos * tensorb->strides[axis] + inner * (axis < rankb - 1 ? tensorb->strides[axis + 1] : 1);
                            size_t dst_offset = outer * result->strides[axis] * (a_axis_elements + b_axis_elements) + (a_axis_elements + b_pos) * result->strides[axis] + inner * (axis < ranka - 1 ? result->strides[axis + 1] : 1);
                            
                            void* elem = b_data + b_src_offset * get_dtype_size(typeb);
							
							// Convert element to int32
                            dst[dst_offset] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
                }
				
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_type);
                nnl2_free_tensor(result);
                return NULL;
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
 * @brief Backend implementations for concat operation
 * @details
 * Array follows the common backend registration pattern for concatenation 
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_concat
 */
Implementation concat_backends[] = {
	REGISTER_BACKEND(naive_concat, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for concat operation
 * @ingroup backend_system 
 */
concatfn nnl2_concat;

/** 
 * @brief Makes the concat backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(concat);

/** 
 * @brief Sets the backend for concat operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for concat
 * @see ESET_BACKEND_BY_NAME
 */
void set_concat_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(concat_backends, nnl2_concat, backend_name, current_backend(concat));
}

/** 
 * @brief Gets the name of the active backend for concat operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_concat_backend() {
	return current_backend(concat);
}

/** 
 * @brief Function declaration for getting all available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(concat);

/**
 * @brief Function declaration for getting the number of available concat backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(concat);

#endif /** NNL2_CONCAT_H **/
