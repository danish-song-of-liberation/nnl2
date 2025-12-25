#ifndef NNL2_F64DGETRF_H
#define NNL2_F64DGETRF_H

// NNL2

#ifdef OPENBLAS_AVAILABLE

/** @brief 
 * Computes the LU factorization of a double-precision matrix using LAPACKE dgetrf.
 *
 ** @param order  
 * Matrix storage layout: nnl2RowMajor or nnl2ColMajor
 * 
 ** @param m       
 * Number of rows of input matrix A (m >= 0)
 *
 ** @param n       
 * Number of columns of input matrix A (n >= 0)
 *
 ** @param a   
 * Input/output matrix A of shape (m, n) and dtype FLOAT64
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
 * -1: Type mismatch (A must be FLOAT64, ipiv must be INT32)
 * -2: Invalid order parameter
 * <0: Other LAPACK error (illegal argument at position -info)
 *
 ** @see LAPACKE_dgetrf
 **/
int nnl2_lapacke_f64dgetrf(const nnl2_order order, const int m, const int n, nnl2_tensor* a, const int lda, nnl2_tensor* ipiv) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(a->dtype != FLOAT64) {
            NNL2_ERROR("In function nnl2_lapacke_f64dgetrf, matrix A must be FLOAT64 type");
            return -1;
        }
        
        if(ipiv->dtype != INT32) {
            NNL2_ERROR("In function nnl2_lapacke_f64dgetrf, ipiv must be INT32 type");
            return -1;
        }
    #endif
    
    // Cast data from void* to appropriate types
    nnl2_float64* a_data = (nnl2_float64*)a->data;
    lapack_int* ipiv_data = (lapack_int*)ipiv->data;

    // Convert nnl2 order enum to LAPACK order enum
    lapack_int lapack_layout;
    
    switch(order) {
        case nnl2RowMajor:
            lapack_layout = LAPACK_ROW_MAJOR;
            break;
            
        case nnl2ColMajor:
            lapack_layout = LAPACK_COL_MAJOR;
            break;
            
        default: {
            NNL2_ORDER_ERROR(order);
            return -2;
        }
    }
    
    // Call LAPACKE dgetrf function
    lapack_int info = LAPACKE_dgetrf(
        lapack_layout,  // Matrix layout
        m,              // Number of rows
        n,              // Number of columns
        a_data,         // Input matrix (overwritten)
        lda,            // Leading dimension
        ipiv_data       // Pivot indices
    );
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
        if(info != 0) {
            if(info < 0) {
                NNL2_WARN("nnl2_lapacke_f64dgetrf: argument %d had an illegal value", -info);
            } else {
                NNL2_WARN("nnl2_lapacke_f64dgetrf: U(%d,%d) is exactly zero (singular matrix)", info, info);
            }
        }
    #endif
    
    return (int)info;
}

#endif /** OPENBLAS_AVAILABLE **/

/** @ingroup backend_system
 ** @brief Backend implementations for double-precision LU factorization
 **/
nnl2_runtime_implementation f64dgetrf_backends[] = {
    #ifdef OPENBLAS_AVAILABLE
        REGISTER_BACKEND(nnl2_lapacke_f64dgetrf, nnl2_lapack, LAPACK_BACKEND_NAME),
    #endif
};

/**
 * @brief Function pointer for double-precision LU factorization
 * @ingroup backend_system 
 */
f64dgetrffn nnl2_f64dgetrf;

/** 
 * @brief Sets the backend for double-precision LU factorization
 * @ingroup backend_system
 */
void set_f64dgetrf_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(f64dgetrf_backends, nnl2_f64dgetrf, backend_name);
}

#endif /** NNL2_F64DGETRF_H **/
