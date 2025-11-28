#ifndef NNL2_AD_GEMMVP_BACKWARD_H
#define NNL2_AD_GEMMVP_BACKWARD_H

/** @file nnl2_ad_gemmvp_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for GEMMVP operation (GEMM + Vector Product)
 **/

/** @brief 
 * Computes the gradient of the GEMMVP operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the GEMMVP operation
 *
 ** @param multiplicand 
 * The first input tensor to the GEMMVP operation (matrix A)
 *
 ** @param multiplier 
 * The second input tensor to the GEMMVP operation (matrix B)
 *
 ** @param vector
 * The bias vector tensor to the GEMMVP operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Computes gradients for: C = A × B + V
 * - dC/dA = gradient @ B^T
 * - dC/dB = A^T @ gradient  
 * - dC/dV = sum(gradient, axis=0) [sum over rows for bias vector]
 *
 ** @see nnl2_gemmvp
 ** @see nnl2_add_inplace
 ** @see nnl2_sum_axis0
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_gemmvp(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* multiplicand, nnl2_ad_tensor* multiplier, nnl2_ad_tensor* vector) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_gemmvp, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand, "In function nnl2_ad_reverse_derivative_gemmvp, multiplicand is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier, "In function nnl2_ad_reverse_derivative_gemmvp, multiplier is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(vector, "In function nnl2_ad_reverse_derivative_gemmvp, vector is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_gemmvp, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data, "In function nnl2_ad_reverse_derivative_gemmvp, multiplicand data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data, "In function nnl2_ad_reverse_derivative_gemmvp, multiplier data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(vector->data, "In function nnl2_ad_reverse_derivative_gemmvp, vector data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->grad, "In function nnl2_ad_reverse_derivative_gemmvp, out_tensor grad is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_gemmvp, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplicand->data->shape, "In function nnl2_ad_reverse_derivative_gemmvp, multiplicand shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(multiplier->data->shape, "In function nnl2_ad_reverse_derivative_gemmvp, multiplier shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(vector->data->shape, "In function nnl2_ad_reverse_derivative_gemmvp, vector shape is NULL");
	#endif
    
    // Extract dimensions
    int m = out_tensor->grad->shape[0];  // rows of output gradient
    int n = out_tensor->grad->shape[1];  // cols of output gradient
    int k = multiplicand->data->shape[1]; // inner dimension
    
    // Compute gradient for multiplicand (A): dC/dA = gradient @ B^T
    nnl2_tensor* grad_out_a = gemm(
        nnl2RowMajor,           // order
        nnl2NoTrans,            // transa
        nnl2Trans,              // transb
        m,                      // rows of result
        k,                      // cols of result (same as multiplicand cols)
        n,                      // common dimension
        1.0,                    // alpha
        out_tensor->grad,       // A (gradient)
        n,                      // lda
        multiplier->data,       // B (multiplier)
        n,                      // ldb
        0.0                     // beta
    );
    
    // Compute gradient for multiplier (B): dC/dB = A^T @ gradient
    nnl2_tensor* grad_out_b = gemm(
        nnl2RowMajor,           // order
        nnl2Trans,              // transa
        nnl2NoTrans,            // transb
        k,                      // rows of result (same as multiplier rows)
        n,                      // cols of result (same as multiplier cols)
        m,                      // common dimension
        1.0,                    // alpha
        multiplicand->data,     // A (multiplicand)
        k,                      // lda
        out_tensor->grad,       // B (gradient)
        n,                      // ldb
        0.0                     // beta
    );
    
    // Compute gradient for bias vector (V): dC/dV = sum(gradient, axis=0)
    // Sum over rows to get gradient for bias vector
    nnl2_tensor* grad_out_v = nnl2_sum_with_axis(out_tensor->grad, 0, false);
    
    // Accumulate gradients if tensors require gradients
    if(multiplicand->requires_grad) nnl2_add_inplace(multiplicand->grad, grad_out_a);
    if(multiplier->requires_grad) nnl2_add_inplace(multiplier->grad, grad_out_b);
    if(vector->requires_grad) nnl2_add_inplace(vector->grad, grad_out_v);
    
    // Free temporary tensors
    nnl2_free_tensor(grad_out_a);
    nnl2_free_tensor(grad_out_b);
    nnl2_free_tensor(grad_out_v);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_GEMMVP_BACKWARD_H **/
