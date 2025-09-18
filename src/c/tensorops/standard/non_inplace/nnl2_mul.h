#ifndef NNL2_MUL_H
#define NNL2_MUL_H

/** @brief
 * Performs element-wise multiplication of two tensors (naive implementation)
 *
 ** @details
 * The function creates a new tensor containing the product of the corresponding elements
 * of the two input tensors. It supports various data types with automatic
 * casting to a higher type in the hierarchy
 *
 ** @param multiplicand
 * Pointer to the multiplicand tensor
 *
 ** @param multiplier
 * Pointer to the multiplier tensor
 *
 ** @return 
 * Pointer to a new tensor with the multiplication result
 *
 ** @note
 * Uses volatile pointers to prevent compiler optimizations
 *
 ** @note
 * Returns NULL in case of failure
 *
 ** @see nnl2_empty
 ** @see get_dtype_size()
 ** @see nnl2_convert_to_float64()
 ** @see nnl2_convert_to_float32()
 ** @see nnl2_convert_to_int32()
 **/
Tensor* nnl2_naive_mul(const Tensor* multiplicand, const Tensor* multiplier) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Calculate the total number of elements in the tensors
    size_t len = product(multiplicand->shape, multiplicand->rank);
    
    TensorType dtype_multiplicand = multiplicand->dtype;
    TensorType dtype_multiplier = multiplier->dtype;
    
    // Selecting the winning type (higher in the hierarchy)
    TensorType winner_in_the_type_hierarchy = MAX(dtype_multiplicand, dtype_multiplier);

    // Create an output tensor with the same shape and winning data type
    Tensor* product = nnl2_empty(multiplicand->shape, multiplicand->rank, winner_in_the_type_hierarchy);
    
    if (product == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return NULL;
    }
    
    if (len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif
        return product;
    }
    
    if (dtype_multiplicand == dtype_multiplier) {
        // Handling the case if the data types match
        
        switch (dtype_multiplicand) {
            case FLOAT64: {
                volatile double* data_multiplicand = (double*)multiplicand->data;
                volatile double* data_multiplier = (double*)multiplier->data;
                volatile double* data_product = (double*)product->data;
            
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_multiplicand = (float*)multiplicand->data;
                volatile float* data_multiplier = (float*)multiplier->data;
                volatile float* data_product = (float*)product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            case INT32: {
                volatile int32_t* data_multiplicand = (int32_t*)multiplicand->data;
                volatile int32_t* data_multiplier = (int32_t*)multiplier->data;
                volatile int32_t* data_product = (int32_t*)product->data;
        
                // Element-wise multiplication
                for (size_t i = 0; i < len; i++) {
                    data_product[i] = data_multiplicand[i] * data_multiplier[i];
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_multiplicand);
                return NULL;
            }
        }
    } else {
        // Handling the case if the data types are NOT match
        switch (winner_in_the_type_hierarchy) {
            case FLOAT64: {
                volatile double* data_product = (double*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float64(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float64(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            case FLOAT32: {
                volatile float* data_product = (float*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_float32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_float32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
        
            case INT32: {
                volatile int32_t* data_product = (int32_t*)product->data;
                
                for (size_t i = 0; i < len; i++) {
                    // Calculate the pointers to the current elements, taking into account the size of the type
                    void* elem_multiplicand = (char*)multiplicand->data + i * get_dtype_size(dtype_multiplicand);
                    void* elem_multiplier = (char*)multiplier->data + i * get_dtype_size(dtype_multiplier);
                    
                    data_product[i] = nnl2_convert_to_int32(elem_multiplicand, dtype_multiplicand) * 
                                     nnl2_convert_to_int32(elem_multiplier, dtype_multiplier);
                }
                
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(winner_in_the_type_hierarchy);
                return NULL;
            }
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return product;
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for multiplication operation
 * @details
 * Array follows the common backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mul: Basic reference implementation
 * 
 * @see nnl2_naive_mul
 */
Implementation mul_backends[] = {
	REGISTER_BACKEND(nnl2_naive_mul, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for multiplication operation
 * @ingroup backend_system 
 */
mulfn mul;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
make_current_backend(mul);

/** 
 * @brief Sets the backend for multiplication operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mul_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mul_backends, mul, backend_name, CURRENT_BACKEND(mul));
}

/** 
 * @brief Gets the name of the active backend for multiplication operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mul_backend() {
	return CURRENT_BACKEND(mul);
}

/** 
 * @brief Function declaration for getting all `mul` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mul);

/**
 * @brief Function declaration for getting the number of all `mul` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mul);

#endif /** NNL2_MUL_H **/
