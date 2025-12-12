#ifndef NNL2_FNN_FORWARD_H
#define NNL2_FNN_FORWARD_H 

// NNL2

/** @file nnl2_fnn_forward.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains FNN Forward function
 **
 *
 * Contains a function for performing direct forward propagation 
 * for a fully connected network. 
 *
 * If use_bias == true, then gemmvp (gemm + bias) is used, 
 * otherwise gemm (general matrix multiplication)
 *
 ** Filepath: src/c/nn/ann/fnn/nnl2_fnn_forward.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Performs forward pass through a Feedforward Neural Network (FNN) layer
 * using weight matrix and bias vector
 *
 ** @param nn 
 * Pointer to the FNN structure containing weights, biases, and metadata
 *
 ** @param x 
 * Input tensor for the forward pass
 *
 ** @return nnl2_ad_tensor* 
 * Output tensor after forward propagation, or NULL on failure
 *
 ** @see nnl2_ad_gemmvp
 ** @see nnl2_nn_fnn
 **/
nnl2_ad_tensor* fnn_forward_with_bias(nnl2_nn_fnn* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function fnn_forward_with_bias_wrapped, nnl2_nn_fnn* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function fnn_forward_with_bias_wrapped, nnl2_ad_tensor* x is NULL", NULL);
    #endif

    nnl2_ad_tensor* out = nnl2_ad_gemmvp(x, nn->weights, nn->bias, nnl2_ad_reverse_mode, true);

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(out, "In function fnn_forward_with_bias_wrapped, nnl2_ad_gemmvp returned NULL", NULL);
    #endif

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return out;
}

/** @brief 
 * Performs forward pass through a Feedforward Neural Network (FNN) layer
 * using weight matrix only (bias disabled)
 *
 ** @param nn 
 * Pointer to the FNN structure containing weights and metadata
 *
 ** @param x 
 * Input tensor for the forward pass
 *
 ** @return nnl2_ad_tensor* 
 * Output tensor after forward propagation, or NULL on failure
 *
 ** @see nnl2_ad_gemm
 ** @see nnl2_nn_fnn
 **/
nnl2_ad_tensor* fnn_forward_no_bias(nnl2_nn_fnn* nn, nnl2_ad_tensor* x) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(nn, "In function fnn_forward_no_bias_wrapped, nnl2_nn_fnn* nn is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(x, "In function fnn_forward_no_bias_wrapped, nnl2_ad_tensor* x is NULL", NULL);
    #endif

    nnl2_ad_tensor* out = nnl2_ad_gemm(x, nn->weights, nnl2_ad_reverse_mode, true);

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(out, "In function fnn_forward_no_bias_wrapped, nnl2_ad_gemm returned NULL", NULL);
    #endif

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif

    return out;
}

#endif /** NNL2_FNN_FORWARD_H **/
