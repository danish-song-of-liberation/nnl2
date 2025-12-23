#ifndef NNL2_F32SGESDD_H
#define NNL2_F32SGESDD_H

// NNL2

#ifdef OPENBLAS_AVAILABLE

/** @brief 
 * Computes the Singular Value Decomposition (SVD) of a single-precision matrix 
 * using LAPACKE sgesdd (Divide-and-Conquer algorithm).
 *
 ** @param order  
 * Matrix storage layout: nnl2RowMajor or nnl2ColMajor
 * 
 ** @param jobz   
 * Specifies options for computing singular vectors:
 * 'A': All m columns of U and all n rows of V^T are returned
 * 'S': First min(m,n) columns of U and rows of V^T are returned
 * 'O': 
 *    1. If jobz = 'O' and m >= n: First n columns of U overwrite A, V^T is computed
 *    2. If jobz = 'O' and m < n: First m rows of V^T overwrite A, U is computed
 *
 * 'N': Neither U nor V^T are computed
 *
 ** @param m       
 * Number of rows of input matrix A (m >= 0)
 *
 ** @param n       
 * Number of columns of input matrix A (n >= 0)
 *
 ** @param a   
 * Input matrix A of shape (m, n) and dtype FLOAT32
 *                   
 ** @param lda     
 * Leading dimension of A (>= max(1,m) for col-major, >= max(1,n) for row-major)
 *
 ** @param s      
 * Output tensor for singular values (size min(m,n), FLOAT32)
 * Singular values are sorted in descending order
 *
 ** @param u      
 * Output tensor for left singular vectors U:
 * If jobz = 'A': shape (m, m)
 * If jobz = 'S': shape (m, min(m,n))
 * If jobz = 'O' and m >= n: shape (m, n) overwrites A
 * If jobz = 'N': not referenced
 * Must be FLOAT32 type
 *
 ** @param ldu     
 * Leading dimension of U:
 * If jobz = 'A': ldu >= max(1,m)
 * If jobz = 'S': ldu >= max(1,m)
 * If jobz = 'O' and m >= n: ldu >= max(1,m)
 * If jobz = 'N': not referenced
 *
 ** @param vt     
 * Output tensor for right singular vectors V^T:
 * If jobz = 'A': shape (n, n)
 * If jobz = 'S': shape (min(m,n), n)
 * If jobz = 'O' and m < n: shape (n, m) overwrites A
 * If jobz = 'N': not referenced
 * Must be FLOAT32 type
 *
 ** @param ldvt    
 * Leading dimension of VT:
 * If jobz = 'A': ldvt >= max(1,n)
 * If jobz = 'S': ldvt >= max(1,min(m,n))
 * If jobz = 'O' and m < n: ldvt >= max(1,n)
 * If jobz = 'N': not referenced
 *
 ** @param iwork
 * Workspace integer tensor of size 8*min(m,n) (INT32)
 *
 ** @return int
 * Integer status code:
 *  0: Success
 * >0: The algorithm failed to converge
 * -1: Type mismatch (all float tensors must be FLOAT32, iwork must be INT32)
 * -2: Invalid jobz parameter
 * -3: Invalid order parameter
 * -4: iwork size insufficient (must be >= 8*min(m,n))
 * <0: Other LAPACK error (illegal argument at position -info)
 *
 ** @note 
 * sgesdd uses divide-and-conquer algorithm which is typically faster than sgesvd
 * for large matrices, but may use more workspace memory.
 *
 ** @note 
 * Memory layout requirements:
 * For column-major: 
 *     If jobz = 'A': ldu >= max(1,m), ldvt >= max(1,n)
 *     If jobz = 'S': ldu >= max(1,m), ldvt >= max(1,min(m,n))
 * For row-major:
 *     If jobz = 'A': ldu >= max(1,m), ldvt >= max(1,n)
 *     If jobz = 'S': ldu >= max(1,m), ldvt >= max(1,min(m,n))
 *
 ** @warning 
 * When jobz = 'O', the input matrix A is overwritten:
 * If m >= n: A is overwritten with the first n columns of U
 * If m < n: A is overwritten with the first m rows of V^T
 *
 ** @example
 * // Compute full SVD of a 1000×500 matrix using divide-and-conquer
 * nnl2_tensor *A, *S, *U, *VT, *iwork;
 *
 * // Initialize tensors with proper dimensions...
 * // iwork must have size 8*min(1000,500) = 8*500 = 4000
 *
 * int info = nnl2_lapacke_f32sgesdd(
 *         nnl2ColMajor, 'A', 
 *         1000, 500, A, 1000,    // A is 1000×500, lda=1000
 *         S,                     // S size = min(1000,500)=500
 *         U, 1000,               // U is 1000×1000, ldu=1000
 *         VT, 500,               // VT is 500×500, ldvt=500
 *         iwork                  // iwork size = 4000
 * );
 *
 ** @see LAPACKE_sgesdd
 ** @see LAPACKE_sgesvd
 ** @see LAPACKE_dgesdd
 ** @see LAPACKE_dgesvd
 **/
int nnl2_lapacke_f32sgesdd(const nnl2_order order, const char jobz,
                           const int m, const int n, nnl2_tensor* a, const int lda,
                           nnl2_tensor* s, nnl2_tensor* u, const int ldu,
                           nnl2_tensor* vt, const int ldvt, nnl2_tensor* iwork) {

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif

    int min_mn = m < n ? m : n;

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        // Check tensor data types
        if(a->dtype != FLOAT32 || s->dtype != FLOAT32 ||
            u->dtype != FLOAT32 || vt->dtype != FLOAT32) {
            NNL2_ERROR("In function nnl2_lapacke_f32sgesdd, float tensors must be FLOAT32 type");
            return -1;
        }
        
        if(iwork->dtype != INT32) {
            NNL2_ERROR("In function nnl2_lapacke_f32sgesdd, iwork tensor must be INT32 type");
            return -1;
        }
    
        // Check jobz parameter
        if(!(jobz == 'A' || jobz == 'S' || jobz == 'O' || jobz == 'N')) {
            NNL2_ERROR("Invalid jobz parameter. Must be 'A', 'S', 'O', or 'N'");
            return -2;
        }
		
		size_t iwork_numel = nnl2_product(iwork->shape, iwork->rank);
        
        // Check iwork size
        if(iwork_numel < 8 * (size_t)min_mn) {
            NNL2_ERROR("iwork size insufficient: need >= %d elements, got %d", 8 * min_mn, iwork_numel);			  
            return -4;
        }
    #endif
    
    // Cast data from void* to appropriate types
    nnl2_float32* a_data = (nnl2_float32*)a->data;
    nnl2_float32* s_data = (nnl2_float32*)s->data;
    nnl2_float32* u_data = (nnl2_float32*)u->data;
    nnl2_float32* vt_data = (nnl2_float32*)vt->data;
    lapack_int* iwork_data = (lapack_int*)iwork->data;

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
    
    lapack_int lwork = -1;
    nnl2_float32 work_query;
    lapack_int info_query = LAPACKE_sgesdd_work(
        lapack_layout,
        jobz,
        m,
        n,
        a_data,
        lda,
        s_data,
        u_data,
        ldu,
        vt_data,
        ldvt,
        &work_query,
        lwork,
        iwork_data
    );
    
    if (info_query != 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
            NNL2_WARN("nnl2_lapacke_f32sgesdd: workspace query failed with info = %d", info_query);
        #endif
        return (int)info_query;
    }
    
    lwork = (lapack_int)work_query;
    if (lwork < 1) {
        int max_mn = m > n ? m : n;
        lwork = 3 * min_mn * min_mn + 4 * min_mn + max_mn;
    }
    
    nnl2_float32* work = (nnl2_float32*)malloc(lwork * sizeof(nnl2_float32));
    if(work == NULL) {
        NNL2_ERROR("Failed to allocate workspace for sgesdd");
        return -5;
    }
    
    // Call LAPACKE sgesdd function
    // Computes the singular value decomposition using divide-and-conquer
    lapack_int info = LAPACKE_sgesdd_work(
        lapack_layout,  // Matrix layout (RowMajor/ColMajor)
        jobz,           // Job option for singular vectors
        m,              // Number of rows of A
        n,              // Number of columns of A
        a_data,         // Input matrix A (may be overwritten)
        lda,            // Leading dimension of A
        s_data,         // Output singular values (size min(m,n))
        u_data,         // Output left singular vectors U
        ldu,            // Leading dimension of U
        vt_data,        // Output right singular vectors V^T
        ldvt,           // Leading dimension of VT
        work,           // Workspace array (float)
        lwork,          // Size of workspace
        iwork_data      // Integer workspace array
    );
    
    // Free workspace
    free(work);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
        if(info != 0) {
            if(info < 0) {
                NNL2_WARN("nnl2_lapacke_f32sgesdd: argument %d had an illegal value", -info);
            } else {
                NNL2_WARN("nnl2_lapacke_f32sgesdd: SBDSDC did not converge, updating process failed");
            }
        }
    #endif
    
    return (int)info;
}

#endif /** OPENBLAS_AVAILABLE **/

/** @ingroup backend_system
 ** @brief Backend implementations for SVD (Divide-and-Conquer) single-precision
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_lapacke_f32sgesdd: LAPACK-accelerated implementation using divide-and-conquer
 *
 ** @see nnl2_lapacke_f32sgesdd
 **/
nnl2_runtime_implementation f32sgesdd_backends[] = {
    #ifdef OPENBLAS_AVAILABLE
        REGISTER_BACKEND(nnl2_lapacke_f32sgesdd, nnl2_lapack, LAPACK_BACKEND_NAME),
    #endif
};

/**
 * @brief Function pointer for SVD (Divide-and-Conquer) single-precision
 * @ingroup backend_system 
 */
f32sgesddfn nnl2_f32sgesdd;

/** 
 * @brief Sets the backend for SVD (Divide-and-Conquer) single-precision
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_f32sgesdd_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(f32sgesdd_backends, nnl2_f32sgesdd, backend_name);
}

#endif /** NNL2_F32SGESDD_H **/
