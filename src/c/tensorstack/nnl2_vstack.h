#ifndef NNL2_VSTACK_H
#define NNL2_VSTACK_H

/** @brief
 * Performs vertical stacking of two tensors (naive implementation)
 *
 ** @param tensora 
 * Pointer to the first input tensor
 *
 ** @param tensorb
 * Pointer to the second input tensor
 *
 ** @return 
 * Pointer to a new tensor containing the vertically stacked result
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* naive_vstack(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    
    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    
    TensorType winner_type = MAX(typea, typeb);
    
	// Calculate total number of elements in each tensor
    size_t sizea = nnl2_product(tensora->shape, tensora->rank);
    size_t sizeb = nnl2_product(tensorb->shape, tensorb->rank);
    
    void* dataa = tensora->data;
    void* datab = tensorb->data;
    
    int* shapea = tensora->shape;
    int* shapeb = tensorb->shape;
    
    size_t shapea_0 = (size_t)shapea[0];
    size_t shapea_1 = (size_t)(ranka > 1 ? shapea[1] : 0);
	
    size_t shapeb_0 = (size_t)shapeb[0];
    size_t shapeb_1 = (size_t)(rankb > 1 ? shapeb[1] : 0);
    
    Tensor* result = NULL;
    
	// Handle 1D-1D case
    if(ranka == 1 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Vectors must have same length for vertical stacking
            if(shapea_0 != shapeb_0) {
                NNL2_ERROR("Vectors must have same length for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [2, vector_length]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = 2;
        shapec[1] = (int)shapea_0;
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec); 
        
        if(typea == typeb && typea == winner_type) {
            // Fast path: same types
            size_t item_size = get_dtype_size(winner_type);
            size_t total_size_a = sizea * item_size;
            size_t total_size_b = sizeb * item_size;
            
            memcpy(result->data, dataa, total_size_a);
            memcpy((char*)result->data + total_size_a, datab, total_size_b);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert and copy first vector
                    for(size_t i = 0; i < shapea_0; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second vector
                    for(size_t i = 0; i < shapeb_0; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[shapea_0 + i] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 
    
	// Handle 2D-1D case
    else if(ranka == 2 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(shapea_1 != shapeb_0) {
                NNL2_ERROR("Matrix columns must match vector length for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [rows+1, columns]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = (int)shapea_0 + 1;
        shapec[1] = (int)shapea_1;
        
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec);
        
        if(typea == typeb && typea == winner_type) {
            // Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size = shapea_1 * item_size;
            
			// Copy matrix data
            memcpy(result->data, dataa, shapea_0 * row_size);
			
			// Append vector as new row
            memcpy((char*)result->data + shapea_0 * row_size, datab, row_size);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Copy and convert matrix
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typea);
                        }
                    }
                    
                    // Copy and convert vector
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Process matrix rows with type conversion
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typea);
                        }
                    }
                    
					// Process vector with type conversion
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
					// Convert matrix elements to int32
                    for(size_t i = 0; i < shapea_0; i++) {
                        for(size_t j = 0; j < shapea_1; j++) {
                            size_t src_idx = i * shapea_1 + j;
                            size_t dst_idx = i * shapea_1 + j;
                            
                            void* elem = (char*)dataa + src_idx * get_dtype_size(typea);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typea);
                        }
                    }
                    
					// Convert vector elements to int32
                    for(size_t j = 0; j < shapeb_0; j++) {
                        size_t dst_idx = shapea_0 * shapea_1 + j;
                        void* elem = (char*)datab + j * get_dtype_size(typeb);
                        dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 
	
	// Handle 1D-2D case
    else if(ranka == 1 && rankb == 2) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			// Vector length must match matrix columns
            if(shapea_0 != shapeb_1) {
                NNL2_ERROR("Vector length must match matrix columns for vstack");
                return NULL;
            }
        #endif
        
		// Allocate memory for result shape [rows+1, columns]
        int* shapec = malloc(2 * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = (int)shapeb_0 + 1;
        shapec[1] = (int)shapeb_1;
        
        result = nnl2_empty(shapec, 2, winner_type);
        free(shapec);
        
        if(typea == typeb && typea == winner_type) {
            // Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t row_size = shapeb_1 * item_size;
            
			// Copy vector as first row
            memcpy(result->data, dataa, row_size);
			
			// Copy matrix data after the vector
            memcpy((char*)result->data + row_size, datab, shapeb_0 * row_size);
        } else {
            // Type conversion needed
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Copy and convert vector as first row
                    for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Copy and convert matrix rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float64(elem, typeb);
                        }
                    }
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert vector to float32 for first row
                    for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert matrix to float32 for remaining rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_float32(elem, typeb);
                        }
                    }
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
                    // Convert vector to int32 for first row
					for(size_t j = 0; j < shapea_0; j++) {
                        void* elem = (char*)dataa + j * get_dtype_size(typea);
                        dst[j] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert matrix to int32 for remaining rows
                    for(size_t i = 0; i < shapeb_0; i++) {
                        for(size_t j = 0; j < shapeb_1; j++) {
                            size_t src_idx = i * shapeb_1 + j;
                            size_t dst_idx = (i + 1) * shapeb_1 + j;
                            
                            void* elem = (char*)datab + src_idx * get_dtype_size(typeb);
                            dst[dst_idx] = nnl2_convert_to_int32(elem, typeb);
                        }
                    }
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
                    return NULL;
                }
            }
        }
    } 

	// Handle general ND-ND case
    else {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN	
            if(ranka != rankb) {
                NNL2_ERROR("Tensors must have same rank for general vstack");
                return NULL;
            }
            
			// All dimensions except axis=0 must match
            for(int i = 1; i < ranka; i++) {
                if(shapea[i] != shapeb[i]) {
                    NNL2_ERROR("Tensors must have same dimensions except axis=0 for vstack");
                    return NULL;
                }
            }
        #endif
        
		// Allocate memory for result shape
        int* shapec = malloc(ranka * sizeof(int));
        
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
		// Concatenate along axis=0, keep other dimensions
        shapec[0] = (int)(shapea_0 + shapeb_0);
        
        for(int i = 1; i < ranka; i++) {
            shapec[i] = shapea[i];
        }

        result = nnl2_empty(shapec, ranka, winner_type);
        free(shapec); 
        
        if(typea == typeb && typea == winner_type) {
			// Same types
            size_t item_size = get_dtype_size(winner_type);
            size_t total_size_a = sizea * item_size;
            size_t total_size_b = sizeb * item_size;
			
			// Copy first tensor
            memcpy(result->data, dataa, total_size_a); 
			
			// Append second tensor
            memcpy((char*)result->data + total_size_a, datab, total_size_b);
        } else {
			// Type conversion needed 
            switch(winner_type) {
                case FLOAT64: {
                    volatile double* dst = (double*)result->data;
                    
                    // Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float64(elem, typea);
                    }
                    
                    // Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_float64(elem, typeb);
                    }
					
                    break;
                }
                
                case FLOAT32: {
                    volatile float* dst = (float*)result->data;
                    
					// Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_float32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_float32(elem, typeb);
                    }
					
                    break;
                }
                
                case INT32: {
                    volatile int32_t* dst = (int32_t*)result->data;
                    
			    	// Convert and copy first tensor
                    for(size_t i = 0; i < sizea; i++) {
                        void* elem = (char*)dataa + i * get_dtype_size(typea);
                        dst[i] = nnl2_convert_to_int32(elem, typea);
                    }
                    
					// Convert and copy second tensor
                    for(size_t i = 0; i < sizeb; i++) {
                        void* elem = (char*)datab + i * get_dtype_size(typeb);
                        dst[sizea + i] = nnl2_convert_to_int32(elem, typeb);
                    }
					
                    break;
                }
                
                default: {
                    NNL2_TYPE_ERROR(winner_type);
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

#ifdef NNL2_PTHREAD_AVAILABLE

/** @brief
 * Threshold for enabling parallel execution of vstack operation
 */
#define NNL2_VSTACK_PARALLEL_THRESHOLD 100000

/** @brief
 * VStack case type identifiers
 */
#define VSTACK_CASE_1D_1D     0
#define VSTACK_CASE_2D_1D     1
#define VSTACK_CASE_1D_2D     2
#define VSTACK_CASE_ND_ND     3

/** @brief
 * Worker function for parallel vstack with same data types
 * 
 ** @param arg 
 * Pointer to vstack_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pvstack_same_type(void* arg);

/** @brief
 * Worker function for parallel vstack with type conversion
 * 
 ** @param arg 
 * Pointer to vstack_ptask structure containing thread parameters
 *
 ** @return 
 * NULL (for pthread API compatibility)
 */
void* nnl2_own_pvstack_convert(void* arg);

/** @brief
 * High-performance parallel implementation of vertical tensor stacking
 * 
 ** @param tensora 
 * Pointer to first source tensor
 *
 ** @param tensorb
 * Pointer to second source tensor
 * 
 ** @return
 * Pointer to new stacked tensor, or NULL if error occurs
 *
 ** @details
 * Uses multi-threading with pthread and optimized memory copying for
 * maximum performance on modern CPU architectures. Handles all vstack
 * cases (1D-1D, 2D-1D, 1D-2D, ND-ND) with appropriate optimizations.
 * 
 ** @note
 * Uses NNL2_NUM_THREADS for parallelization configuration
 * Falls back to naive implementation for small tensors or complex cases
 * 
 ** @warning
 * Requires pthread support
 */
Tensor* nnl2_own_vstack(const Tensor* tensora, const Tensor* tensorb) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora, "First tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb, "Second tensor is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensora->data, "First tensor data is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(tensorb->data, "Second tensor data is NULL", NULL);
    #endif
    
    TensorType typea = tensora->dtype;
    TensorType typeb = tensorb->dtype;
    int ranka = tensora->rank;
    int rankb = tensorb->rank;
    TensorType winner_type = MAX(typea, typeb);
    bool same_type = (typea == typeb && typea == winner_type);

    size_t sizea = nnl2_product(tensora->shape, tensora->rank);
    size_t sizeb = nnl2_product(tensorb->shape, tensorb->rank);
    size_t total_elements = sizea + sizeb;

    // Fallback to naive implementation for small tensors
    if (total_elements < NNL2_VSTACK_PARALLEL_THRESHOLD) {
        return naive_vstack(tensora, tensorb);
    }

    Tensor* result = NULL;
    int case_type = VSTACK_CASE_ND_ND;
    
    // Determine vstack case and create result tensor
    if(ranka == 1 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(tensora->shape[0] != tensorb->shape[0]) {
                NNL2_ERROR("Vectors must have same length for vstack");
                return NULL;
            }
        #endif
        
        int shapec[2] = {2, tensora->shape[0]};
        result = nnl2_empty(shapec, 2, winner_type);
        case_type = VSTACK_CASE_1D_1D;
    } 
    else if(ranka == 2 && rankb == 1) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(tensora->shape[1] != tensorb->shape[0]) {
                NNL2_ERROR("Matrix columns must match vector length for vstack");
                return NULL;
            }
        #endif
        
        int shapec[2] = {tensora->shape[0] + 1, tensora->shape[1]};
        result = nnl2_empty(shapec, 2, winner_type);
        case_type = VSTACK_CASE_2D_1D;
    }
    else if(ranka == 1 && rankb == 2) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(tensora->shape[0] != tensorb->shape[1]) {
                NNL2_ERROR("Vector length must match matrix columns for vstack");
                return NULL;
            }
        #endif
        
        int shapec[2] = {tensorb->shape[0] + 1, tensorb->shape[1]};
        result = nnl2_empty(shapec, 2, winner_type);
        case_type = VSTACK_CASE_1D_2D;
    }
    else {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN	
            if(ranka != rankb) {
                NNL2_ERROR("Tensors must have same rank for general vstack");
                return NULL;
            }
            
            for(int i = 1; i < ranka; i++) {
                if(tensora->shape[i] != tensorb->shape[i]) {
                    NNL2_ERROR("Tensors must have same dimensions except axis=0 for vstack");
                    return NULL;
                }
            }
        #endif
        
        int* shapec = malloc(ranka * sizeof(int));
        if (shapec == NULL) {
            NNL2_ERROR("Memory allocation failed");
            return NULL; 
        }
        
        shapec[0] = tensora->shape[0] + tensorb->shape[0];
        for(int i = 1; i < ranka; i++) {
            shapec[i] = tensora->shape[i];
        }

        result = nnl2_empty(shapec, ranka, winner_type);
        free(shapec);
        case_type = VSTACK_CASE_ND_ND;
    }
    
    if(result == NULL) {
        NNL2_ERROR("Failed to create result tensor in vstack");
        return NULL;
    }

    bool src_a_aligned = NNL2_IS_ALIGNED(tensora->data, NNL2_TENSOR_ALIGNMENT_32);
    bool src_b_aligned = NNL2_IS_ALIGNED(tensorb->data, NNL2_TENSOR_ALIGNMENT_32);
    bool dst_aligned = NNL2_IS_ALIGNED(result->data, NNL2_TENSOR_ALIGNMENT_32);
    bool is_aligned = src_a_aligned && src_b_aligned && dst_aligned;
    
    // Warning for unaligned memory in safety modes
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!is_aligned) {
            NNL2_WARN("In nnl2_own_vstack, tensor memory is not aligned to 32 bytes. Performance may be suboptimal");
        }
    #endif
    
    size_t num_threads = NNL2_NUM_THREADS;
    pthread_t threads[num_threads];
    vstack_ptask tasks[num_threads];
    
    // Calculate row sizes for different cases
    size_t row_size_a = 0, row_size_b = 0;
    size_t item_size = get_dtype_size(winner_type);
    
    switch(case_type) {
        case VSTACK_CASE_1D_1D: {
            row_size_a = (size_t)tensora->shape[0] * item_size;
            row_size_b = (size_t)tensorb->shape[0] * item_size;
            break;
		}
		
        case VSTACK_CASE_2D_1D: {
            row_size_a = (size_t)tensora->shape[1] * item_size;
            row_size_b = row_size_a; // Same row size for vector
            break;
		}
		
        case VSTACK_CASE_1D_2D: {
            row_size_a = (size_t)tensora->shape[0] * item_size;
            row_size_b = (size_t)tensorb->shape[1] * item_size;
            break;
		}
		
        case VSTACK_CASE_ND_ND: {
            row_size_a = sizea * item_size;
            row_size_b = sizeb * item_size;
            break;
		}
    }
    
    // Configure tasks for parallel processing
    size_t chunk = total_elements / num_threads;
    size_t remainder = total_elements % num_threads;
    
    size_t current_start = 0;
    for (size_t i = 0; i < num_threads; i++) {
        size_t current_chunk = chunk + (i < remainder ? 1 : 0);
        
        tasks[i].src_a = tensora->data;
        tasks[i].src_b = tensorb->data;
        tasks[i].dst = result->data;
        tasks[i].start_idx = current_start;
        tasks[i].end_idx = current_start + current_chunk;
        tasks[i].size_a = sizea;
        tasks[i].size_b = sizeb;
        tasks[i].row_size_a = row_size_a;
        tasks[i].row_size_b = row_size_b;
        tasks[i].type_a = typea;
        tasks[i].type_b = typeb;
        tasks[i].result_type = winner_type;
        tasks[i].aligned = is_aligned;
        tasks[i].same_type = same_type;
        tasks[i].case_type = case_type;
        
        current_start += current_chunk;
    }
    
    // Create and run threads
    for (size_t i = 0; i < num_threads; i++) {
        void* (*worker_func)(void*) = tasks[i].same_type ? 
            nnl2_own_pvstack_same_type : nnl2_own_pvstack_convert;
        
        int status = pthread_create(&threads[i], NULL, worker_func, &tasks[i]);
        if(status != 0) {
            NNL2_THREAD_CREATE_ERROR(status, "nnl2_own_vstack");
            num_threads = i;
            break;
        }
    }
    
    // Wait for all threads to complete
    for (size_t i = 0; i < num_threads; i++) {
        int join_status = pthread_join(threads[i], NULL);
        if(join_status != 0) {
            NNL2_THREAD_JOIN_ERROR(join_status, "nnl2_own_vstack");
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

// Worker function implementations

/** @brief
 * Parallel vstack worker for same data types
 * 
 ** @see nnl2_own_pvstack_same_type
 **/
void* nnl2_own_pvstack_same_type(void* arg) {
    vstack_ptask* task = (vstack_ptask*)arg;
    char* src_a = (char*)task->src_a;
    char* src_b = (char*)task->src_b;
    char* dst = (char*)task->dst;
    
    size_t item_size = get_dtype_size(task->result_type);
    size_t total_size_a = task->size_a * item_size;
    
    // Calculate byte ranges for this thread
    size_t start_byte = task->start_idx * item_size;
    size_t end_byte = task->end_idx * item_size;
    
    if (end_byte <= total_size_a) {
        // Entire chunk in first tensor
        memcpy(dst + start_byte, src_a + start_byte, end_byte - start_byte);
    } else if (start_byte >= total_size_a) {
        // Entire chunk in second tensor
        size_t src_start = start_byte - total_size_a;
        size_t src_end = end_byte - total_size_a;
        memcpy(dst + start_byte, src_b + src_start, src_end - src_start);
    } else {
        // Chunk spans both tensors
        size_t first_part = total_size_a - start_byte;
        memcpy(dst + start_byte, src_a + start_byte, first_part);
        memcpy(dst + total_size_a, src_b, end_byte - total_size_a);
    }
    
    return NULL;
}

/** @brief
 * Parallel vstack worker for type conversion
 * 
 ** @see nnl2_own_pvstack_convert
 **/
void* nnl2_own_pvstack_convert(void* arg) {
    vstack_ptask* task = (vstack_ptask*)arg;
    
    // Process elements in this thread's range
    for (size_t i = task->start_idx; i < task->end_idx; i++) {
        if (i < task->size_a) {
            // Element from first tensor
            void* elem = (char*)task->src_a + i * get_dtype_size(task->type_a);
            size_t dst_idx = i;
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float64(elem, task->type_a);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float32(elem, task->type_a);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_int32(elem, task->type_a);
                    break;
                }
				
                default: break;
            }
        } else {
            // Element from second tensor
            size_t src_idx = i - task->size_a;
            void* elem = (char*)task->src_b + src_idx * get_dtype_size(task->type_b);
            size_t dst_idx = i;
            
            switch(task->result_type) {
                case FLOAT64: {
                    double* dst = (double*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float64(elem, task->type_b);
                    break;
                }
				
                case FLOAT32: {
                    float* dst = (float*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_float32(elem, task->type_b);
                    break;
                }
				
                case INT32: {
                    int32_t* dst = (int32_t*)task->dst;
                    dst[dst_idx] = nnl2_convert_to_int32(elem, task->type_b);
                    break;
                } 
				
                default: break;
            }
        }
    }
    
    return NULL;
}

#endif

/**
 * @ingroup backend_system
 * @brief Backend implementations for vstack operation
 * @details
 * Array follows the common backend registration pattern for vertical stacking operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for vertical tensor concatenation
 * 
 * @see nnl2_naive
 * @see naive_vstack
 */
Implementation vstack_backends[] = {
	REGISTER_BACKEND(naive_vstack, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef NNL2_PTHREAD_AVAILABLE
		REGISTER_BACKEND(nnl2_own_vstack, nnl2_own, NNL2_OWN_NAME),
	#endif
};	

/**
 * @brief Function pointer for vstack operation
 * @ingroup backend_system 
 */
vstackfn vstack;

/** 
 * @brief Makes the vstack backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(vstack);

/** 
 * @brief Sets the backend for vstack operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for vertical stacking
 * @see ESET_BACKEND_BY_NAME
 */
void set_vstack_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(vstack_backends, vstack, backend_name, current_backend(vstack));
}

/** 
 * @brief Gets the name of the active backend for vstack operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_vstack_backend() {
	return current_backend(vstack);
}

/** 
 * @brief Function declaration for getting all available vstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(vstack);

/**
 * @brief Function declaration for getting the number of available vstack backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(vstack);

#endif /** NNL2_VSTACK_H **/
