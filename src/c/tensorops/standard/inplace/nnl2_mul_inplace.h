#ifndef NNL2_MUL_INPLACE_H
#define NNL2_MUL_INPLACE_H

/** @brief 
 * Performs element-wise multiplication of two tensors (naive implementation)
 * 
 * Multiplies the elements of the multiplicand tensor by the corresponding elements 
 * of the multiplier tensor, modifying the multiplicand tensor in place
 *
 ** @param multiplicand 
 * Pointer to the tensor that will be modified (receives the multiplication result)
 *
 ** @param multiplier 
 * Pointer to the tensor whose values will multiply the multiplicand
 *
 ** @note 
 * Supports different data types through automatic conversion
 * The multiplier elements are converted to the multiplicand's data type
 *
 ** @note 
 * In max safety mode, validates tensor pointers and data pointers before access
 *
 ** @warning 
 * This function modifies the multiplicand tensor directly
 *
 * @example
 * // Create two tensors with the same shape
 * Tensor* a = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * Tensor* b = nnl2_ones((int[]){2, 3}, 2, FLOAT32);
 * 
 * // Multiply a by b (a becomes a * b)
 * naive_mulinplace(a, b);
 * 
 * // Now a contains 1.0 in all elements
 * nnl2_quick_print_tensor(a);
 * 
 * // Cleanup
 * nnl2_free_tensor(a);
 * nnl2_free_tensor(b);
 */
void nnl2_naive_mulinplace(Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Additional checks at the maximum safety level
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX   
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "Multiplicand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "Multiplicand tensor's data is NULL");
        
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "Multiplier tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "Multiplier tensor's data is NULL");
    #endif
    
    // Calculating the total number of elements in the multiplicand tensor
    size_t len_multiplicand = product(multiplicand->shape, multiplicand->rank);
    
    // If the tensor is empty, exit the function
    if(len_multiplicand == 0) return;
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    if(dtype_multiplicand == dtype_multiplier) {
        // Handling case when the tensors have the same type
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];    
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                
                // Element-wise multiplication
                for(size_t i = 0; i < len_multiplicand; i++) data_multiplicand[i] *= data_multiplier[i];        
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return;
            }
        }
    } else {
        // Handling the case when tensors have different data types
        // Calculating the step size for accessing multiplier tensor elements
        size_t multiplier_step = get_dtype_size(dtype_multiplier);
        
        // Casting multiplier data to char* for byte access
        char* multiplier_data = (char*)multiplier->data;
        
        switch(dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT64 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float64(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                
                // For each element, convert the multiplier element to FLOAT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_float32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                
                // For each element, convert the multiplier element to INT32 and multiply it
                for(size_t i = 0; i < len_multiplicand; i++) {
                    void* multiplier_elem = multiplier_data + i * multiplier_step;
                    data_multiplicand[i] *= nnl2_convert_to_int32(multiplier_elem, dtype_multiplier);
                }
                
                break; 
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mulinplace: Basic reference implementation
 * 
 * @see nnl2_naive_mulinplace
 */
Implementation mulinplace_backends[] = {
	REGISTER_BACKEND(nnl2_naive_mulinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulinplacefn mulinplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(mulinplace);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mulinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mulinplace_backends, mulinplace, backend_name, CURRENT_BACKEND(mulinplace));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mulinplace_backend() {
	return CURRENT_BACKEND(mulinplace);
}

/** 
 * @brief Function declaration for getting all `mulinplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mulinplace);

/**
 * @brief Function declaration for getting the number of all `mulinplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mulinplace);

#endif /** NNL2_MUL_INPLACE_H **/
