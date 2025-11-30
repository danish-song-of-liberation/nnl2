#ifndef NNL2_MSE_H
#define NNL2_MSE_H

// NNL2

/** @brief 
 * Computes Mean Squared Error (MSE) between prediction and target tensors
 *
 ** @param prediction 
 * Pointer to prediction tensor
 *
 ** @param target 
 * Pointer to target tensor (must match prediction shape and dtype)
 *
 ** @param record 
 * Pointer to memory where result will be stored
 */
void nnl2_naive_mse(nnl2_tensor* prediction, nnl2_tensor* target, void* record) {
    size_t numel = product(prediction -> shape, prediction -> rank);
    
    switch(prediction -> dtype) {
        case FLOAT64: {
            nnl2_float64 acc = 0.0;
            nnl2_float64* prediction_data = (nnl2_float64*)prediction -> data;
            nnl2_float64* target_data = (nnl2_float64*)target -> data;
            
            for(size_t it = 0; it < numel; it++) {
                nnl2_float64 diff = prediction_data[it] - target_data[it];
				
				// Faster that math.h pow
                acc += diff * diff;  
            }
            
            nnl2_float64* result = (nnl2_float64*)record;
            *result = (numel > 0) ? acc / (nnl2_float64)numel : 0.0;
            break;
        }
        
        case FLOAT32: {
            nnl2_float32 acc = 0.0f;
            nnl2_float32* prediction_data = (nnl2_float32*)prediction -> data;
            nnl2_float32* target_data = (nnl2_float32*)target -> data;
            
            for(size_t it = 0; it < numel; it++) {
                nnl2_float32 diff = prediction_data[it] - target_data[it];
                acc += diff * diff;
            }
            
            nnl2_float32* result = (nnl2_float32*)record;
            *result = (numel > 0) ? acc / (nnl2_float32)numel : 0.0f;
            break;
        }
        
        case INT32: {
            int64_t acc = 0; 
            nnl2_int32* prediction_data = (nnl2_int32*)prediction -> data;
            nnl2_int32* target_data = (nnl2_int32*)target -> data;
            
            for(size_t it = 0; it < numel; it++) {
                nnl2_int32 diff = prediction_data[it] - target_data[it];
                acc += (int64_t)diff * diff;
            }
            
            // Always use FLOAT64 for INT32 inputs
            nnl2_float64* result = (nnl2_float64*)record;
            *result = (numel > 0) ? (nnl2_float64)acc / (nnl2_float64)numel : 0.0;
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(prediction -> dtype);
            break;
        }
    }
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for MSE loss operation
 * @details
 * Array follows the standard backend registration pattern.
 * Currently registered backends:
 *  - nnl2_naive_mse: Basic reference implementation
 *
 * @see nnl2_naive_mse
 */
Implementation mse_backends[] = {
    REGISTER_BACKEND(nnl2_naive_mse, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for MSE loss operation
 * @ingroup backend_system
 */
msefn nnl2_mse;

/**
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(mse);

/**
 * @brief Sets the backend for MSE loss operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_mse_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(mse_backends, nnl2_mse, backend_name, CURRENT_BACKEND(mse));
}

/**
 * @brief Gets the name of the active backend for MSE loss operation
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_mse_backend() {
    return CURRENT_BACKEND(mse);
}

/**
 * @brief Function declaration for getting all `mse` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(mse);

/**
 * @brief Function declaration for getting the number of all `mse` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(mse);

#endif /** NNL2_MSE_H **/
