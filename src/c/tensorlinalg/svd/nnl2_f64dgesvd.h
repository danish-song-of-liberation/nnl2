#ifndef NNL2_F64DGESVD_H
#define NNL2_F64DGESVD_H

// NNL2

#ifdef OPENBLAS_AVAILABLE

/** @brief 
 * Computes the Singular Value Decomposition (SVD) of a double-precision matrix using LAPACKE dgesvd.
 *
 ** @param order  
 * Matrix storage layout: nnl2RowMajor or nnl2ColMajor
 * 
 ** @param jobu    
 * Specifies options for computing U:
 * 'A': All m columns of U are returned
 * 'S': First min(m,n) columns of U are returned
 * 'O': First min(m,n) columns of U overwrite A
 * 'N': U is not computed
 *
 ** @param jobvt   
 * Specifies options for computing V^T:
 * 'A': All n rows of V^T are returned
 * 'S': First min(m,n) rows of V^T are returned
 * 'O': First min(m,n) rows of V^T overwrite A
 * 'N': V^T is not computed
 * 
 ** @param m       
 * Number of rows of input matrix A (m >= 0)
 *
 ** @param n       
 * Number of columns of input matrix A (n >= 0)
 *
 ** @param a   
 * Input matrix A of shape (m, n) and dtype FLOAT64
 *                   
 ** @param lda     
 * Leading dimension of A (>= max(1,m) for col-major, >= max(1,n) for row-major)
 *
 ** @param s      
 * Output tensor for singular values (size min(m,n), FLOAT64)
 *      
 ** @param u      
 * Output tensor for left singular vectors U:
 * If jobu = 'A': shape (m, m)
 * If jobu = 'S': shape (m, min(m,n))
 * If jobu = 'N': not referenced
 * Must be FLOAT64 type
 *
 ** @param ldu     
 * Leading dimension of U (>= max(1,m) for col-major if jobu != 'N')
 *
 ** @param vt     
 * Output tensor for right singular vectors V^T:
 * If jobvt = 'A': shape (n, n)
 * If jobvt = 'S': shape (min(m,n), n)
 * If jobvt = 'N': not referenced
 * Must be FLOAT64 type
 *
 ** @param ldvt    
 * Leading dimension of VT (>= max(1,n) for col-major if jobvt != 'N')
 *
 ** @param superb 
 * Workspace tensor of size min(m,n)-1 (FLOAT64)
 *
 ** @return int
 * Integer status code:
 *  0: Success
 * >0: Number of superdiagonals that failed to converge
 * -1: Type mismatch (all tensors must be FLOAT64)
 * -2: Invalid jobu or jobvt parameter
 * -3: Invalid order parameter
 * <0: Other LAPACK error (illegal argument at position -info)
 *
 ** @note 
 * Memory layout requirements:
 * For column-major: lda ≥ max(1,m), ldu ≥ max(1,m), ldvt ≥ max(1,n)
 * For row-major: lda ≥ max(1,n), ldu ≥ max(1,m), ldvt ≥ max(1,n)
 *
 ** @warning 
 * If jobu='O' and jobvt='O', A is overwritten and both U and VT are not computed
 *
 ** @example
 * // Compute full SVD of a 3×2 matrix
 * nnl2_tensor *A, *S, *U, *VT, *superb;
 *
 * // Initialize tensors with proper dimensions...
 *
 * int info = nnl2_lapacke_f64dgesvd(
 *         nnl2ColMajor, 'A', 'A', 
 *         3, 2, A, 3,    // A is 3×2, lda=3
 *         S,             // S size = min(3,2)=2
 *         U, 3,          // U is 3×3, ldu=3
 *         VT, 2,         // VT is 2×2, ldvt=2
 *         superb         // superb size = 1
 * );
 *
 ** @see LAPACKE_dgesvd
 ** @see LAPACKE_dgesdd
 ** @see LAPACKE_sgesvd
 ** @see LAPACKE_sgesdd
 **/
int nnl2_lapacke_f64dgesvd(const nnl2_order order, const char jobu, const char jobvt,
						   const int m, const int n, nnl2_tensor* a, const int lda,
						   nnl2_tensor* s, nnl2_tensor* u, const int ldu,
					       nnl2_tensor* vt, const int ldvt, nnl2_tensor* superb) {

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif

	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(a->dtype != FLOAT64 || s->dtype != FLOAT64 ||
			u->dtype != FLOAT64 || vt->dtype != FLOAT64 ||
			superb->dtype != FLOAT64) {
			NNL2_ERROR("In function nnl2_lapacke_f64dgesvd, all tensors must be FLOAT64 type for dgesvd");
			return -1;
		}
	
		if(!(jobu == 'A' || jobu == 'S' || jobu == 'O' || jobu == 'N') ||
			!(jobvt == 'A' || jobvt == 'S' || jobvt == 'O' || jobvt == 'N')) {
			NNL2_ERROR("Invalid jobu or jobvt parameter. Must be 'A', 'S', 'O', or 'N'");
			return -2;
		}
	#endif
	
	// Cast data from void* to double*
    nnl2_float64* a_data = (nnl2_float64*)a->data;
    nnl2_float64* s_data = (nnl2_float64*)s->data;
    nnl2_float64* u_data = (nnl2_float64*)u->data;
    nnl2_float64* vt_data = (nnl2_float64*)vt->data;
    nnl2_float64* superb_data = (nnl2_float64*)superb->data;

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
            return -3;
        }
    }
	
	// Call LAPACKE dgesvd function
    // Computes the singular value decomposition of a general rectangular matrix
    lapack_int info = LAPACKE_dgesvd(
        lapack_layout,  // Matrix layout (RowMajor/ColMajor)
        jobu,           // Job option for U
        jobvt,          // Job option for V^T
        m,              // Number of rows of A
        n,              // Number of columns of A
        a_data,         // Input matrix A (may be overwritten)
        lda,            // Leading dimension of A
        s_data,         // Output singular values (size min(m,n))
        u_data,         // Output left singular vectors U
        ldu,            // Leading dimension of U
        vt_data,        // Output right singular vectors V^T
        ldvt,           // Leading dimension of VT
        superb_data     // Workspace array (size min(m,n)-1)
    );
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
        if(info != 0) {
            if(info < 0) {
                NNL2_WARN("nnl2_lapacke_f64dgesvd: argument %d had an illegal value", -info);
            } else {
                NNL2_WARN("nnl2_lapacke_f64dgesvd: %d superdiagonals failed to converge", info);
            }
        }
    #endif
	
	return (int)info;
}

#endif /** OPENBLAS_AVAILABLE **/

/** @ingroup backend_system
 ** @brief Backend implementations for SVD
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_lapacke_f64svd: LAPACK-accelerated implementation
 *
 ** @see nnl2_lapacke_f64svd
 **/
nnl2_runtime_implementation f64dgesvd_backends[] = {
    #ifdef OPENBLAS_AVAILABLE
        REGISTER_BACKEND(nnl2_lapacke_f64dgesvd, nnl2_lapack, LAPACK_BACKEND_NAME),
    #endif
};

/**
 * @brief Function pointer for SVD
 * @ingroup backend_system 
 */
f64dgesvdfn nnl2_f64dgesvd;

/** 
 * @brief Sets the backend for SVD
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_f64dgesvd_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(f64dgesvd_backends, nnl2_f64dgesvd, backend_name);
}

#endif /** NNL2_F64DGESVD_H **/
