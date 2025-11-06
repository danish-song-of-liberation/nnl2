#ifndef NNL2_AD_GEMM_BACKWARD_H
#define NNL2_AD_GEMM_BACKWARD_H

/** @file nnl2_ad_gemm_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for GEMM operation
 **/

/** @brief 
 * Computes the gradient of the GEMM operation in reverse mode AD
 *
 ** @param out_tensor 
 * The output tensor from the GEMM operation
 *
 ** @param addend 
 * The first input tensor to the GEMM operation
 *
 ** @param sumend 
 * The second input tensor to the GEMM operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * MVP
 *
 ** @see gemm
 ** @see nnl2_add_inplace
 ** @see nnl2_free_tensor
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_gemm(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor, "In function nnl2_ad_reverse_derivative_gemm, out_tensor is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend, "In function nnl2_ad_reverse_derivative_gemm, addend is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "In function nnl2_ad_reverse_derivative_gemm, sumend is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data, "In function nnl2_ad_reverse_derivative_gemm, out_tensor data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data, "In function nnl2_ad_reverse_derivative_gemm, addend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "In function nnl2_ad_reverse_derivative_gemm, sumend data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(out_tensor->data->shape, "In function nnl2_ad_reverse_derivative_gemm, out_tensor shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(addend->data->shape, "In function nnl2_ad_reverse_derivative_gemm, addend shape is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data->shape, "In function nnl2_ad_reverse_derivative_gemm, sumend shape is NULL");
	#endif
    
    int m = out_tensor->grad->shape[0];
    int k = out_tensor->grad->shape[1];
    int n = sumend->data->shape[1];
    
    nnl2_tensor* grad_out_a = gemm(
        nnl2RowMajor, 		     // order
        nnl2NoTrans,    		 // transa
        nnl2Trans,          	 // transb
        m, 					     // rows of result
        addend->data->shape[1],  // cols of result  
        k, 						 // common dimension
        1.0, 					 // alpha
        out_tensor->grad,        // A
        k, 						 // lda
        sumend->data,		     // B
        n,  					 // ldb
        0.0  					 // beta
    );
    
    nnl2_tensor* grad_out_b = gemm(
        nnl2RowMajor,   		 // order
        nnl2Trans,  	    	 // transa
        nnl2NoTrans, 		     // transb
        addend->data->shape[0],  // rows of result
        n,  					 // cols of result
        m,  					 // common dimension
        1.0, 					 // alpha
        addend->data, 			 // A
        k,  					 // lda  
        out_tensor->grad, 		 // B
        n,  					 // ldb
        0.0  					 // beta
    );
    
    if(addend->requires_grad) nnl2_add_inplace(addend->grad, grad_out_a);
    if(sumend->requires_grad) nnl2_add_inplace(sumend->grad, grad_out_b);
	
    nnl2_free_tensor(grad_out_a);
    nnl2_free_tensor(grad_out_b);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
}

#endif /** NNL2_AD_GEMM_BACKWARD_H **/
