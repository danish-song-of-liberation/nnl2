#ifndef NNL2_NN_RELU_H
#define NNL2_NN_RELU_H

/** @file nnl2_nn_relu.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains common ReLU structure for neural networks
 **/
 
 

///@{ [nnl2_nn_relu]

typedef struct nnl2_nn_relu_struct {
    nnl2_nn_ann metadata;  ///< Base neural network metadata
} nnl2_nn_relu;

///@} [nnl2_nn_relu]



/** @brief 
 * Creates a ReLU activation layer
 *
 ** @return nnl2_nn_relu* 
 * A pointer to the newly created ReLU activation layer
 *
 ** @retval NULL
 * Returned if memory allocation fails
 *
 ** @warning
 * The caller is responsible for freeing the memory by calling
 * `nnl2_nn_relu_free()` on the returned pointer
 *
 ** @see nnl2_nn_relu_free
 ** @see nnl2_nn_relu_forward
 ** @see nnl2_nn_relu_get_parameters
 **/
nnl2_nn_relu* nnl2_nn_relu_create(void) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_relu* nn = malloc(sizeof(nnl2_nn_relu));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    // Metadata
    nn -> metadata.nn_type = nnl2_nn_type_relu;
    nn -> metadata.use_bias = false; 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

/** @brief 
 * Destroys a ReLU activation layer and releases its memory
 *
 ** @param nn 
 * Pointer to the ReLU layer to be destroyed
 *
 ** @note 
 * Safe to call with NULL pointer (does nothing)
 * ReLU has no trainable parameters, so only metadata is freed
 *
 ** @see nnl2_nn_relu_create
 **/
void nnl2_nn_relu_free(nnl2_nn_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(nn, "In function nnl2_nn_relu_free, nnl2_nn_relu* nn is NULL");
    #endif 
    
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves trainable parameters from a ReLU activation layer
 *
 ** @param nn 
 * Pointer to the ReLU layer
 *
 ** @return 
 * Always returns NULL 
 *
 ** @see nnl2_nn_relu_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_relu_get_parameters(nnl2_nn_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_relu_get_parameters, nnl2_nn_relu* nn is NULL", NULL);
    #endif 
    
    nnl2_ad_tensor** params = NULL;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

/** @brief 
 * Returns the number of trainable parameters in a ReLU activation layer (0)
 *
 ** @param nn 
 * Pointer to the ReLU layer
 *
 ** @return 
 * Always returns 0
 *
 ** @see nnl2_nn_relu_get_parameters
 **/
size_t nnl2_nn_relu_get_num_parameters(nnl2_nn_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_relu_get_num_parameters, nnl2_nn_relu* nn is NULL", 0);
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 0;  
}

/** @brief 
 * Performs forward pass through a ReLU activation layer
 *
 ** @param nn 
 * Pointer to the ReLU layer
 *
 ** @param x 
 * Input tensor to apply ReLU activation to
 *
 ** @see nnl2_nn_relu_create
 ** @see nnl2_ad_relu
 **/
nnl2_ad_tensor* nnl2_nn_relu_forward(nnl2_nn_relu* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_relu_forward, nnl2_nn_relu* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function nnl2_nn_relu_forward, nnl2_ad_tensor* x is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* forward_pass = nnl2_ad_relu(x, nnl2_ad_reverse_mode);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_relu_forward, failed to compute ReLU activation", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

#endif /** NNL2_NN_RELU_H **/
