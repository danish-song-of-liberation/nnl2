#ifndef NNL2_NN_SIGMOID_H
#define NNL2_NN_SIGMOID_H

/** @file nnl2_nn_sigmoid.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains common sigmoid structure for neural networks
 **/
 
 

///@{ [nnl2_nn_sigmoid]

typedef struct nnl2_nn_sigmoid_struct {
    nnl2_nn_ann metadata;  ///< Base neural network metadata
	bool approx;		   ///< Approximated sigmoid for performance
} nnl2_nn_sigmoid;

///@} [nnl2_nn_sigmoid]



/** @brief 
 * Creates a Sigmoid activation layer
 *
 ** @param approx
 * If true, uses an approximate fast sigmoid implementation with lower precision
 * but better performance
 *
 ** @return nnl2_nn_sigmoid* 
 * A pointer to the newly created Sigmoid activation layer
 *
 ** @retval NULL
 * Returned if memory allocation fails
 *
 ** @warning
 * The caller is responsible for freeing the memory by calling
 * `nnl2_nn_sigmoid_free()` on the returned pointer
 *
 ** @see nnl2_nn_sigmoid_free
 ** @see nnl2_nn_sigmoid_forward
 ** @see nnl2_nn_sigmoid_get_parameters
 **/
nnl2_nn_sigmoid* nnl2_nn_sigmoid_create(bool approx) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_sigmoid* nn = malloc(sizeof(nnl2_nn_sigmoid));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!nn) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }   
    #endif
    
    // Metadata
    nn -> metadata.nn_type = nnl2_nn_type_sigmoid;
    nn -> metadata.use_bias = false; 
	
	// Common data
	nn -> approx = approx;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return nn;
}

/** @brief 
 * Destroys a Sigmoid activation layer and releases its memory
 *
 ** @param nn 
 * Pointer to the Sigmoid layer to be destroyed
 *
 ** @note 
 * Safe to call with NULL pointer (does nothing)
 * Sigmoid has no trainable parameters, so only metadata is freed
 *
 ** @see nnl2_nn_sigmoid_create
 **/
void nnl2_nn_sigmoid_free(nnl2_nn_sigmoid* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(nn, "In function nnl2_nn_sigmoid_free, nnl2_nn_sigmoid* nn is NULL");
    #endif 
    
    free(nn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves trainable parameters from a Sigmoid activation layer
 *
 ** @param nn 
 * Pointer to the Sigmoid layer
 *
 ** @return 
 * Always returns NULL 
 *
 ** @see nnl2_nn_sigmoid_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_sigmoid_get_parameters(nnl2_nn_sigmoid* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_sigmoid_get_parameters, nnl2_nn_sigmoid* nn is NULL", NULL);
    #endif 
    
    nnl2_ad_tensor** params = NULL;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return params; 
}

/** @brief 
 * Returns the number of trainable parameters in a sigmoid activation layer (0)
 *
 ** @param nn 
 * Pointer to the sigmoid layer
 *
 ** @return 
 * Always returns 0
 *
 ** @see nnl2_nn_sigmoid_get_parameters
 **/
size_t nnl2_nn_sigmoid_get_num_parameters(nnl2_nn_sigmoid* nn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_sigmoid_get_num_parameters, nnl2_nn_sigmoid* nn is NULL", 0);
    #endif 
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return 0;  
}

/** @brief 
 * Performs forward pass through a Sigmoid activation layer
 *
 ** @param nn 
 * Pointer to the Sigmoid layer
 *
 ** @param x 
 * Input tensor to apply sigmoid activation to
 *
 ** @see nnl2_nn_sigmoid_create
 ** @see nnl2_ad_sigmoid
 **/
nnl2_ad_tensor* nnl2_nn_sigmoid_forward(nnl2_nn_sigmoid* nn, nnl2_ad_tensor* x) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function nnl2_nn_sigmoid_forward, nnl2_nn_sigmoid* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function nnl2_nn_sigmoid_forward, nnl2_ad_tensor* x is NULL", NULL);
    #endif
	
	nnl2_ad_tensor* forward_pass = nnl2_ad_sigmoid(x, nn -> approx, nnl2_ad_reverse_mode);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(forward_pass, "In function nnl2_nn_sigmoid_forward, failed to compute sigmoid activation", NULL);
    #endif
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return forward_pass;
}

/** @brief 
 * Print sigmoid activation layer information
 *
 ** @param nn 
 * Pointer to the sigmoid layer structure
 *
 ** @param terpri 
 * If true, print a newline after the output. Needs for sequential
 */
void nnl2_nn_sigmoid_print(nnl2_nn_sigmoid* nn, bool terpri) {
    if(!nn) {
        printf("(.sigmoid NULL)%s", terpri ? "\n" : "");
        return;
    }
    
    printf("(.sigmoid :approx %s)%s", nn->approx ? "t" : "nil", terpri ? "\n" : "");
}

#endif /** NNL2_NN_SIGMOID_H **/
