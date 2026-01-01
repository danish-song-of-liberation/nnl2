#ifndef NNL2_NN_SOFTMAX_H
#define NNL2_NN_SOFTMAX_H

/** @file nnl2_nn_softmax.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains common softmax structure for neural networks
 **/
 
 

///@{ [nnl2_nn_softmax]

typedef struct nnl2_nn_softmax_struct {
    nnl2_nn_ann metadata;  ///< Base neural network metadata
    int dim;               ///< Dimension along which softmax is applied (default: -1 for last dimension)
} nnl2_nn_softmax;

///@} [nnl2_nn_softmax]



/** @brief 
 * Creates a Softmax activation layer
 *
 ** @param dim
 * Dimension along which softmax will be applied
 * Use -1 for automatic detection (last dimension)
 *
 ** @return nnl2_nn_softmax* 
 * A pointer to the newly created Softmax activation layer
 *
 ** @retval NULL
 * Returned if memory allocation fails
 *
 ** @warning
 * The caller is responsible for freeing the memory by calling
 * `nnl2_nn_softmax_free()` on the returned pointer
 *
 ** @see nnl2_nn_softmax_free
 ** @see nnl2_nn_softmax_forward
 ** @see nnl2_nn_softmax_get_parameters
 **/
nnl2_nn_softmax* nnl2_nn_softmax_create(int dim) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_softmax* nn = malloc(sizeof(nnl2_nn_softmax));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    // Metadata
    nn -> metadata.nn_type = nnl2_nn_type_softmax;
    nn -> metadata.use_bias = false; 
    nn -> metadata.nn_magic = NNL2_NN_MAGIC;
    
    // Softmax specific
    nn -> dim = dim;  // -1 means last dimension
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

/** @brief 
 * Destroys a Softmax activation layer and releases its memory
 *
 ** @param nn 
 * Pointer to the Softmax layer to be destroyed
 *
 ** @note 
 * Safe to call with NULL pointer (does nothing)
 * Softmax has no trainable parameters, so only metadata is freed
 *
 ** @see nnl2_nn_softmax_create
 **/
void nnl2_nn_softmax_free(nnl2_nn_softmax* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return;
        }
    #endif 
    
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves trainable parameters from a Softmax activation layer
 *
 ** @param nn 
 * Pointer to the Softmax layer
 *
 ** @return 
 * Always returns NULL 
 *
 ** @see nnl2_nn_softmax_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_softmax_get_parameters(nnl2_nn_softmax* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return NULL;
        }
    #endif 
    
    nnl2_ad_tensor** params = NULL;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

/** @brief 
 * Returns the number of trainable parameters in a Softmax activation layer (0)
 *
 ** @param nn 
 * Pointer to the Softmax layer
 *
 ** @return 
 * Always returns 0
 *
 ** @see nnl2_nn_softmax_get_parameters
 **/
size_t nnl2_nn_softmax_get_num_parameters(nnl2_nn_softmax* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return 0;
        }
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 0;  
}

/** @brief 
 * Gets the dimension along which softmax is applied
 *
 ** @param nn 
 * Pointer to the Softmax layer
 *
 ** @return 
 * The dimension index (-1 for last dimension)
 *
 ** @see nnl2_nn_softmax_create
 **/
int nnl2_nn_softmax_get_dim(nnl2_nn_softmax* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return -1;
        }
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn->dim;
}

/** @brief 
 * Sets the dimension along which softmax is applied
 *
 ** @param nn 
 * Pointer to the Softmax layer
 *
 ** @param dim
 * The dimension index (-1 for last dimension)
 *
 ** @see nnl2_nn_softmax_create
 **/
void nnl2_nn_softmax_set_dim(nnl2_nn_softmax* nn, int dim) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            return;
        }
    #endif 
    
    nn->dim = dim;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Performs forward pass through a Softmax activation layer
 *
 ** @param nn 
 * Pointer to the Softmax layer
 *
 ** @param x 
 * Input tensor to apply softmax activation to
 *
 ** @param track_graph
 * Whether to track this operation in the computation graph for backpropagation
 *
 ** @return nnl2_ad_tensor*
 * Output tensor after softmax activation, or NULL on failure
 *
 ** @exception NNL2Error
 * Returns NULL if nn or x is NULL
 *
 ** @exception NNL2Error
 * Returns NULL if softmax operation fails
 *
 ** @see nnl2_nn_softmax_create
 ** @see nnl2_ad_softmax
 **/
nnl2_ad_tensor* nnl2_nn_softmax_forward(nnl2_nn_softmax* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_softmax_forward, nnl2_nn_softmax* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function nnl2_nn_softmax_forward, nnl2_ad_tensor* x is NULL", NULL);
    #endif
    
    int dim = nn->dim;
    if(dim < 0) {
        // use last dimension
        dim = x->data->rank - 1;
    }

    nnl2_ad_tensor* forward_pass = nnl2_ad_softmax(x, dim, nnl2_ad_reverse_mode, true);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_softmax_forward, failed to compute softmax activation", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

#endif /** NNL2_NN_SOFTMAX_H **/
