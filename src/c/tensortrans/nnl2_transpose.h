#ifndef NNL2_TRANSPOSE_H
#define NNL2_TRANSPOSE_H

/** @brief 
 * Performs 2D tensor transposition using a naive method
 *
 ** @param tensor 
 * Pointer to the original tensor
 *
 ** @param force
 * If true - performs full transposition with data reordering
 * If false - only swaps dimensions without reordering data (copy with swapped shape)
 *
 ** @return
 * Tensor* Pointer to a transposed tensor
 *
 ** @see https://en.wikipedia.org/wiki/Dont_repeat_yourself 
 **/
Tensor* naive_transpose(Tensor* tensor, bool force) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Make sure the tensor is 2D
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        int32_t tensor_rank = tensor->rank;
        if(tensor_rank < 2 || tensor_rank > 2) {
            NNL2_FATAL("Tensor have an incorrect rank: %d (expected 2)", tensor_rank);
            return NULL;
        }
    #endif
    
    // Create swapped shape for the transposed tensor
    int32_t transposed_shape[2];
    transposed_shape[0] = tensor->shape[1];  // cols become rows
    transposed_shape[1] = tensor->shape[0];  // rows become cols
    
    // Create a tensor result with swapped dimensions
    Tensor* result = nnl2_empty(transposed_shape, tensor->rank, tensor->dtype);
    if (result == NULL) {
        NNL2_FATAL("Failed to initialize the tensor in transposition");
        return NULL;
    }

    int rows = tensor->shape[0];
    int cols = tensor->shape[1];
    
    if (force) {
        // Full transposition with data reordering
        switch(tensor->dtype) {
            case FLOAT64: {
                double* src_data = (double*)tensor->data;
                double* dest_data = (double*)result->data;

                // Transposition: element [i][j] becomes [j][i]
                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        int orig_index = i * cols + j;   // Index in the original matrix
                        int trans_index = j * rows + i;  // Index in the transposed matrix

                        dest_data[trans_index] = src_data[orig_index];
                    }
                }
                break;
            }
            
            case FLOAT32: {
                float* src_data = (float*)tensor->data;
                float* dest_data = (float*)result->data;

                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        int orig_index = i * cols + j;
                        int trans_index = j * rows + i;
                        dest_data[trans_index] = src_data[orig_index];
                    }
                }
                break;
            }
            
            case INT32: {
                int32_t* src_data = (int32_t*)tensor->data;
                int32_t* dest_data = (int32_t*)result->data;

                for(int i = 0; i < rows; i++) {
                    for(int j = 0; j < cols; j++) {
                        int orig_index = i * cols + j;
                        int trans_index = j * rows + i;
                        dest_data[trans_index] = src_data[orig_index];
                    }
                }
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                return NULL;
            }
        }
    } else {
        // Only swap dimensions - copy data in original order
        size_t data_size;
        switch(tensor->dtype) {
            case FLOAT64: data_size = sizeof(double); break;
            case FLOAT32: data_size = sizeof(float); break;
            case INT32: data_size = sizeof(int32_t); break;
            default: {
                NNL2_TYPE_ERROR(tensor->dtype);
                return NULL;
            }
        }
        
        // Calculate total number of elements
        size_t total_elements = rows * cols;
        
        // Copy data as-is without reordering
        memcpy(result->data, tensor->data, total_elements * data_size);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transpose operation
 * @details
 * Array follows the common backend registration pattern for transpose
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for transpose
 * 
 * @see nnl2_naive
 * @see naive_transpose
 */
Implementation transpose_backends[] = {
    REGISTER_BACKEND(naive_transpose, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transpose operation
 * @ingroup backend_system 
 */
transposefn transpose;

/** 
 * @brief Makes the transpose backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(transpose);

/** 
 * @brief Sets the backend for transpose operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transpose
 * @see ESET_BACKEND_BY_NAME
 */
void set_transpose_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transpose_backends, transpose, backend_name, CURRENT_BACKEND(transpose));
}

/** 
 * @brief Gets the name of the active backend for transpose operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transpose_backend() {
    return CURRENT_BACKEND(transpose);
}

/** 
 * @brief Function declaration for getting all available transpose backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transpose);

/**
 * @brief Function declaration for getting the number of available transpose backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transpose);

#endif /** NNL2_TRANSPOSE_H **/
