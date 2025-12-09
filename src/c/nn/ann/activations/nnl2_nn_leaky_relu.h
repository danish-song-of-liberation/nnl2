#ifndef NNL2_NN_LEAKY_RELU_H
#define NNL2_NN_LEAKY_RELU_H

/** @file nnl2_nn_leaky_relu.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains common .leaky-relu structure for neural networks
 **/
 
 

///@{ [nnl2_nn_leakyrelu]

typedef struct nnl2_nn_leaky_relu_struct {
    nnl2_nn_ann metadata;  ///< Base neural network metadata
    float alpha;           ///< Slope for negative values
} nnl2_nn_leaky_relu;

///@} [nnl2_nn_leakyrelu]



/** @brief 
 * Creates a .leaky-relu activation layer
 *
 ** @param alpha
 * Slope for negative values (typically 0.01). Must be > 0
 *
 ** @return nnl2_nn_leaky_relu* 
 * A pointer to the newly created .leaky-relu activation layer
 *
 ** @retval NULL
 * Returned if memory allocation fails
 *
 ** @warning
 * The caller is responsible for freeing the memory by calling
 * `nnl2_nn_leaky_relu_free()` on the returned pointer
 *
 ** @see nnl2_nn_leaky_relu_free
 ** @see nnl2_nn_leaky_relu_forward
 ** @see nnl2_nn_leaky_relu_get_parameters
 **/
nnl2_nn_leaky_relu* nnl2_nn_leaky_relu_create(float alpha) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_leaky_relu* nn = malloc(sizeof(nnl2_nn_leaky_relu));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    // Metadata
    nn -> metadata.nn_type = nnl2_nn_type_leaky_relu;
    nn -> metadata.use_bias = false; 
    
    // Common data
    nn -> alpha = alpha;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

/** @brief 
 * Destroys a .leaky-relu activation layer and releases its memory
 *
 ** @param nn 
 * Pointer to the .leaky-relu layer to be destroyed
 *
 ** @note 
 * Safe to call with NULL pointer (does nothing)
 * .leaky-relu has no trainable parameters, so only metadata is freed
 *
 ** @see nnl2_nn_leaky_relu_create
 **/
void nnl2_nn_leaky_relu_free(nnl2_nn_leaky_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(nn, "In function nnl2_nn_leaky_relu_free, nnl2_nn_leaky_relu* nn is NULL");
    #endif 
    
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves trainable parameters from a .leaky-relu activation layer
 *
 ** @param nn 
 * Pointer to the .leaky-relu layer
 *
 ** @return 
 * Always returns NULL 
 *
 ** @see nnl2_nn_leaky_relu_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_leaky_relu_get_parameters(nnl2_nn_leaky_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_leaky_relu_get_parameters, nnl2_nn_leaky_relu* nn is NULL", NULL);
    #endif 
    
    nnl2_ad_tensor** params = NULL;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

/** @brief 
 * Returns the number of trainable parameters in a .leaky-relu activation layer (0)
 *
 ** @param nn 
 * Pointer to the .leaky-relu layer
 *
 ** @return 
 * Always returns 0
 *
 ** @see nnl2_nn_leaky_relu_get_parameters
 **/
size_t nnl2_nn_leaky_relu_get_num_parameters(nnl2_nn_leaky_relu* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_leaky_relu_get_num_parameters, nnl2_nn_leaky_relu* nn is NULL", 0);
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 0;  
}

/** @brief 
 * Performs forward pass through a .leaky-relu activation layer
 *
 ** @param nn 
 * Pointer to the .leaky-relu layer
 *
 ** @param x 
 * Input tensor to apply .leaky-relu activation to
 *
 ** @see nnl2_nn_leaky_relu_create
 ** @see nnl2_ad_leaky_relu
 **/
nnl2_ad_tensor* nnl2_nn_leaky_relu_forward(nnl2_nn_leaky_relu* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_leaky_relu_forward, nnl2_nn_leaky_relu* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function nnl2_nn_leaky_relu_forward, nnl2_ad_tensor* x is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* forward_pass = nnl2_ad_leakyrelu(x, nn -> alpha, false, nnl2_ad_reverse_mode);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_leaky_relu_forward, failed to compute .leaky-relu activation", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

#endif /** NNL2_NN_LEAKY_RELU_H **/
