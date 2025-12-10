#ifndef NNL2_NN_TANH_H
#define NNL2_NN_TANH_H

/** @file nnl2_nn_tanh.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains common tanh structure for neural networks
 **/
 
 

///@{ [nnl2_nn_tanh]

typedef struct nnl2_nn_tanh_struct {
    nnl2_nn_ann metadata;  ///< Base neural network metadata
	bool approx;           ///< Approximated tanh for performance
} nnl2_nn_tanh;

///@} [nnl2_nn_tanh]



/** @brief 
 * Creates a Tanh activation layer
 *
 ** @param approx
 * If true, uses an approximate fast tanh implementation with lower precision
 * but better performance
 *
 ** @return nnl2_nn_tanh* 
 * A pointer to the newly created Tanh activation layer
 *
 ** @retval NULL
 * Returned if memory allocation fails
 *
 ** @warning
 * The caller is responsible for freeing the memory by calling
 * `nnl2_nn_tanh_free()` on the returned pointer
 *
 ** @see nnl2_nn_tanh_free
 ** @see nnl2_nn_tanh_forward
 ** @see nnl2_nn_tanh_get_parameters
 **/
nnl2_nn_tanh* nnl2_nn_tanh_create(bool approx) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_tanh* nn = malloc(sizeof(nnl2_nn_tanh));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    // Metadata
    nn -> metadata.nn_type = nnl2_nn_type_tanh;
    nn -> metadata.use_bias = false; 
    
    // Common data
    nn -> approx = approx;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

/** @brief 
 * Destroys a Tanh activation layer and releases its memory
 *
 ** @param nn 
 * Pointer to the Tanh layer to be destroyed
 *
 ** @note 
 * Safe to call with NULL pointer (does nothing)
 * Tanh has no trainable parameters, so only metadata is freed
 *
 ** @see nnl2_nn_tanh_create
 **/
void nnl2_nn_tanh_free(nnl2_nn_tanh* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(nn, "In function nnl2_nn_tanh_free, nnl2_nn_tanh* nn is NULL");
    #endif 
    
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves trainable parameters from a Tanh activation layer
 *
 ** @param nn 
 * Pointer to the Tanh layer
 *
 ** @return 
 * Always returns NULL 
 *
 ** @see nnl2_nn_tanh_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_tanh_get_parameters(nnl2_nn_tanh* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_tanh_get_parameters, nnl2_nn_tanh* nn is NULL", NULL);
    #endif 
    
    nnl2_ad_tensor** params = NULL;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

/** @brief 
 * Returns the number of trainable parameters in a tanh activation layer (0)
 *
 ** @param nn 
 * Pointer to the tanh layer
 *
 ** @return 
 * Always returns 0
 *
 ** @see nnl2_nn_tanh_get_parameters
 **/
size_t nnl2_nn_tanh_get_num_parameters(nnl2_nn_tanh* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_tanh_get_num_parameters, nnl2_nn_tanh* nn is NULL", 0);
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 0;  
}

/** @brief 
 * Performs forward pass through a Tanh activation layer
 *
 ** @param nn 
 * Pointer to the Tanh layer
 *
 ** @param x 
 * Input tensor to apply tanh activation to
 *
 ** @see nnl2_nn_tanh_create
 ** @see nnl2_ad_tanh
 **/
nnl2_ad_tensor* nnl2_nn_tanh_forward(nnl2_nn_tanh* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_tanh_forward, nnl2_nn_tanh* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function nnl2_nn_tanh_forward, nnl2_ad_tensor* x is NULL", NULL);
    #endif
    
    nnl2_ad_tensor* forward_pass = nnl2_ad_tanh(x, nn -> approx, nnl2_ad_reverse_mode);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_tanh_forward, failed to compute tanh activation", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

/** @brief 
 * Print hyperbolic tangent activation layer information
 *
 ** @param nn 
 * Pointer to the tanh layer structure
 ** @param terpri 
 * If true, print a newline after the output. Needs for sequential
 */
void nnl2_nn_tanh_print(nnl2_nn_tanh* nn, bool terpri) {
    if (!nn) {
        printf("(.tanh NULL)%s", terpri ? "\n" : "");
        return;
    }
    
    printf("(.tanh :approx %s)%s", nn->approx ? "t" : "nil", terpri ? "\n" : "");
}

#endif /** NNL2_NN_TANH_H **/
