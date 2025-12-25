#ifndef NNL2_GETRF_H
#define NNL2_GETRF_H

// NNL2

/** @brief 
 * Computes the LU factorization of a general matrix with partial pivoting.
 * This is a unified dispatcher that calls the appropriate precision-specific implementation.
 *
 ** @param order  
 * Matrix storage layout 
 * nnl2RowMajor or nnl2ColMajor
 * 
 ** @param m       
 * Number of rows of input matrix A (m >= 0)
 *
 ** @param n       
 * Number of columns of input matrix A (n >= 0)
 *
 ** @param a   
 * Input/output matrix A of shape (m, n)
 * On exit: contains the factors L and U in compact form
 *                   
 ** @param lda     
 * Leading dimension of A (>= max(1,m) for col-major, >= max(1,n) for row-major)
 *
 ** @param ipiv      
 * Output tensor for pivot indices (size min(m,n), INT32)
 * Pivot indices: for 1 <= i <= min(m,n), row i was interchanged with row ipiv[i-1]
 * Note: LAPACK uses 1-based indexing for ipiv
 *      
 ** @return int
 * Integer status code:
 *  0: Success
 * >0: U(i,i) is exactly zero (matrix is singular)
 * -1: Unsupported data type
 * <0: Other LAPACK error (illegal argument at position -info)
 *
 ** @warning 
 * Matrix A is overwritten with the LU factors in compact form:
 * The upper triangle contains U
 * The lower triangle (excluding diagonal) contains multipliers of L
 * Diagonal of L is implied to be 1.0
 *
 ** @see nnl2_f32sgetrf
 ** @see nnl2_f64dgetrf
 **/
int nnl2_getrf(const nnl2_order order, const int m, const int n, nnl2_tensor* a, const int lda, nnl2_tensor* ipiv) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    // Dispatch based on data type
    switch(a->dtype) {
        case FLOAT64:  return nnl2_f64dgetrf(order, m, n, a, lda, ipiv);
        case FLOAT32:  return nnl2_f32sgetrf(order, m, n, a, lda, ipiv);
            
        default: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MINIMAL
                NNL2_TYPE_ERROR(a->dtype);
            #endif
			
            return -1;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
    
    // Should never reach here
    return -1;
}

#endif /** NNL2_GETRF_H **/
