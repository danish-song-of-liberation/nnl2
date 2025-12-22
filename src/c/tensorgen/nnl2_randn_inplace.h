#ifndef NNL2_RANDN_INPLACE_H
#define NNL2_RANDN_INPLACE_H

#include <math.h>

/** @file nnl2_randn_inplace.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains definition of functions that fills tensor with random values from standard normal distribution N(0, 1) in-place
 **/

/** @brief
 * Fills the given tensor with random values from normal distribution N(mean, std²) (in-place)
 *
 ** @param tensor 
 * Tensor to fill with random values from N(mean, std²)
 *
 ** @param mean
 * Mean of the normal distribution
 *
 ** @param std
 * Standard deviation of the normal distribution
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If tensor is NULL
 *
 ** @exception NNL2Error 
 * If passed tensor with unknown type
 *
 ** @example
 * // Fill a tensor with random floats from N(0, 1)
 * nnl2_tensor* tensor = nnl2_empty((int[]){2, 2}, 2, FLOAT32);
 * nnl2_naive_randn_inplace(tensor, 0.0, 1.0);
 *
 ** @example
 * // Fill a tensor with random floats from N(5.0, 2.0²)
 * nnl2_tensor* tensor = nnl2_empty((int[]){3, 3}, 2, FLOAT64);
 * nnl2_naive_randn_inplace(tensor, 5.0, 2.0);
 *
 ** @see nnl2_randn
 ** @see nnl2_rand_inplace
 **/
void nnl2_naive_randn_inplace(nnl2_tensor* tensor, double mean, double std) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_naive_randn_inplace, passed tensor is NULL");
    #endif
    
    size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
    if(total_elems == 0) return; // If zero elems then early return
    
    // Filling with random values from N(mean, std²)
    switch(tensor->dtype) {
        case FLOAT64: {
            nnl2_float64* data = (nnl2_float64*)tensor->data;
         
            for(size_t i = 0; i + 1 < total_elems; i += 2) {
                double u1 = 1.0 - ((double)rand() / RAND_MAX);
                double u2 = 1.0 - ((double)rand() / RAND_MAX);
                
                double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
                
                data[i] = mean + std * z0;
				
                if(i + 1 < total_elems) 
                    data[i + 1] = mean + std * z1;
            }

            if(total_elems % 2 == 1) {
                double u1 = 1.0 - ((double)rand() / RAND_MAX);
                double u2 = 1.0 - ((double)rand() / RAND_MAX);
                double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
                data[total_elems - 1] = mean + std * z0;
            }
			
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* data = (nnl2_float32*)tensor->data;
            float mean_f = (float)mean;
            float std_f = (float)std;

            for(size_t i = 0; i + 1 < total_elems; i += 2) {
                float u1 = 1.0f - ((float)rand() / RAND_MAX);  
                float u2 = 1.0f - ((float)rand() / RAND_MAX);  
                
                float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
                float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
                
                data[i] = mean_f + std_f * z0;
				
                if(i + 1 < total_elems) 
                    data[i + 1] = mean_f + std_f * z1;
            }
            
            if(total_elems % 2 == 1) {
                float u1 = 1.0f - ((float)rand() / RAND_MAX);
                float u2 = 1.0f - ((float)rand() / RAND_MAX);
                float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
                data[total_elems - 1] = mean_f + std_f * z0;
            }
			
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
 * @brief Backend implementations for randn_inplace operation
 * @details
 * Array follows the common backend registration pattern for standard normal 
 * random number generation operations (in-place version). Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for standard normal distribution (in-place)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_randn_inplace
 */
Implementation randn_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_randn_inplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for randn_inplace operation
 * @ingroup backend_system
 */
randninplacefn randn_inplace;

/** 
 * @brief Makes the randn_inplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(randn_inplace);

/** 
 * @brief Sets the backend for randn_inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for randn_inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_randn_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(randn_inplace_backends, randn_inplace, backend_name, current_backend(randn_inplace));
}

/** 
 * @brief Gets the name of the active backend for randn_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_randn_inplace_backend() {
    return current_backend(randn_inplace);
}

/**
 * @brief Function declaration for getting all available randn_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(randn_inplace);

/**
 * @brief Function declaration for getting the number of available randn_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(randn_inplace);

#endif /** NNL2_RANDN_INPLACE_H **/
