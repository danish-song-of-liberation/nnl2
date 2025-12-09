#ifndef NNL2_ANN_BACKEND_H
#define NNL2_ANN_BACKEND_H 

// NNL2

/** @file nnl2_ann_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Backend types and utilities
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_backend.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/



///@{ [nnl2_nn_type]

typedef enum {
    nnl2_nn_type_fnn,           ///< Fully Connected Neural Network 
	nnl2_nn_type_sequential,    ///< Sequential neural network (layers in sequence)
	nnl2_nn_type_sigmoid,		///< Autologous
    nnl2_nn_type_unknown        ///< Unknown or unsupported network type 
} nnl2_nn_type;

///@} [nnl2_nn_type]



///@{ [nnl2_nn_ann]

typedef struct {
    nnl2_nn_type nn_type;  ///< Type of the neural network 
    bool use_bias;         ///< Whether the network uses bias terms 
} nnl2_nn_ann;

///@} [nnl2_nn_ann]



/** @brief 
 * Retrieves the type of a neural network
 *
 ** @param nn 
 * Pointer to the neural network structure
 *
 ** @return nnl2_nn_type
 * The type of the neural network, or nnl2_nn_type_unknown on error
 *
 ** @see nnl2_nn_type
 **/
nnl2_nn_type nnl2_nn_get_type(void* nn) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_nn_get_type, nn is NULL. returning unknown type (nnl2_nn_type_unknown)");
            #endif
                
            return nnl2_nn_type_unknown;
        }
    #endif
    
    nnl2_nn_type* nn_type_mem = (nnl2_nn_type*)nn;
    nnl2_nn_type result = *nn_type_mem;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_ANN_BACKEND_H **/