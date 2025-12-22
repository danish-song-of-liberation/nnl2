#ifndef NNL2_RANDN_H
#define NNL2_RANDN_H

#include <math.h>

/** @brief
 * Creates a tensor with random numbers from normal distribution N(mean, std²)
 *
 ** @param shape
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank
 * Number of dimensions (length of shape array)
 *
 ** @param dtype
 * Float data type of the tensor elements
 *
 ** @param mean
 * Mean of the normal distribution
 *
 ** @param std
 * Standard deviation of the normal distribution
 *
 ** @return nnl2_tensor*
 * Pointer to the newly created tensor
 *
 ** @example
 * // Create a 2x3 tensor of random floats from N(0, 1)
 * nnl2_tensor* normal_tensor = nnl2_randn((int[]){2, 3}, 2, FLOAT32, 0.0, 1.0);
 *
 ** @see naive_rand
 ** @see nnl2_empty
 **/
nnl2_tensor* naive_randn(int* shape, int rank, TensorType dtype, double mean, double std) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    Tensor* result = nnl2_empty(shape, rank, dtype);
    
    size_t total_elems = nnl2_product(shape, rank);
    if(total_elems == 0) return result;
    
    switch(dtype) {
        case FLOAT64: {
            double* data = (double*)result->data;
			
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
            float* data = (float*)result->data;
            float mean_f = (float)mean;
            float std_f = (float)std;

            for(size_t i = 0; i + 1 < total_elems; i += 2) {
                float u1 = 1.0f - ((float)rand() / RAND_MAX);  
                float u2 = 1.0f - ((float)rand() / RAND_MAX); 
                
                float z0 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
                float z1 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
                
                data[i] = mean_f + std_f * z0;
                if(i + 1 < total_elems) {
                    data[i + 1] = mean_f + std_f * z1;
                }
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
            NNL2_TYPE_ERROR(dtype);
            nnl2_free_tensor(result);
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
 * @brief Backend implementations for randn operation
 * @details
 * Array follows the common backend registration pattern for standard normal 
 * random number generation. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation using Box-Muller transform
 * 
 * @see nnl2_naive
 * @see naive_randn
 */
Implementation randn_backends[] = {
    REGISTER_BACKEND(naive_randn, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for randn operation
 * @ingroup backend_system 
 */
randnfn nnl2_randn;

/** 
 * @brief Makes the randn backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(randn);

/** 
 * @brief Sets the backend for randn operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for randn
 * @see ESET_BACKEND_BY_NAME
 */
void set_randn_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(randn_backends, nnl2_randn, backend_name, current_backend(randn));
}

/** 
 * @brief Gets the name of the active backend for randn operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_randn_backend() {
    return current_backend(randn);
}

/** 
 * @brief Function declaration for getting all available randn backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(randn);

/**
 * @brief Function declaration for getting the number of available randn backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(randn);

#endif /** NNL2_RANDN_H **/
