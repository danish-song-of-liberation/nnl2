#ifndef NNL2_DGEMM_INPLACE_H
#define NNL2_DGEMM_INPLACE_H

/** @brief
 * Naive implementation of double-precision general matrix multiplication (DGEMM) in-place
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * This is a naive triple-loop implementation
 * Not optimized for performance - use for reference only
 *
 ** @note
 * All input tensors must be of FLOAT64 type and properly allocated
 *
 ** @note
 * Matrix dimensions must satisfy: 
 * - A: [m x k] if transa == NoTrans, [k x m] if transa == Trans
 * - B: [k x n] if transb == NoTrans, [n x k] if transb == Trans  
 * - C: [m x n]
 *
 ** @note
 * Leading dimensions must be >= corresponding matrix dimensions
 *
 ** @example
 * // Multiply two matrices: C = alpha * A * B + beta * C
 * naive_dgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @exception
 * Function returns early with error message if invalid parameters are detected
 *
 **/
void naive_dgemminplace(const nnl2_order order, const nnl2_transpose transa,
                        const nnl2_transpose transb, const int m, const int n,
                        const int k, const double alpha, const nnl2_tensor* a, const int lda,
                        const nnl2_tensor* b, const int ldb, const double beta, nnl2_tensor* c,
                        const int ldc) {	

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	

	// Checking the input data for correctness
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (!a || !b || !c || !a->data || !b->data || !c->data) {
			NNL2_ERROR("Null pointer passed as argument (gemm)");
			return;
		}

		if (m <= 0 || n <= 0 || k <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
			NNL2_ERROR("Invalid dimensions provided (gemm)");
			return;
		}

		int a_cols = (transa == nnl2Trans) ? m : k;
		int b_cols = (transb == nnl2Trans) ? k : n;

		if (lda < a_cols) {
			NNL2_ERROR("lda is less than number of columns of a matrix");
			return;
		}

		if (ldb < b_cols) {
			NNL2_ERROR("ldb is less than number of columns of b matrix (gemm)");
			return;
		}

		if (ldc < n) {
			NNL2_ERROR("ldc is less than n (gemm)");
			return;
		}
	#endif

	// Casting data from void*
    volatile double* data_a = (volatile double*)a->data;
    volatile double* data_b = (volatile double*)b->data;
    volatile double* data_c = (volatile double*)c->data;

    if(order == nnl2RowMajor){
		// nnl2_runtime_implementation for RowMajor order (row-wise data organization)
		
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                double acc = 0.0;

                for(volatile int l = 0; l < k; l++) {
                    double a_val;
                    double b_val;

                    if (transa == nnl2Trans) {
                        a_val = *(data_a + l * lda + i);
                    } else {
                        a_val = *(data_a + i * lda + l);
                    }

                    if (transb == nnl2Trans) {
                        b_val = *(data_b + j * ldb + l);
                    } else {
                        b_val = *(data_b + l * ldb + j);
                    }

                    acc += a_val * b_val;
                }

                if(beta == 0) {
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    } else {
		// nnl2_runtime_implementation for ColumnMajor order (column-wise data organization)
		
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                double acc = 0.0;

                for(volatile int l = 0; l < k; l++) {
                    double a_val;
                    double b_val;

                    if (transa == nnl2Trans) {
                        a_val = *(data_a + i * lda + l);
                    } else {
                        a_val = *(data_a + l * lda + i);
                    }

                    if (transb == nnl2Trans) {
                        b_val = *(data_b + l * ldb + j);
                    } else {
                        b_val = *(data_b + j * ldb + l);
                    }

                    acc += a_val * b_val;
                }

                if(beta == 0) {
					// C[i,j] = alpha * (A * B)[i,j]
                    *(data_c + i * ldc + j) = acc * alpha;
                } else {
					// C[i,j] = alpha * (A * B)[i,j] + beta * C[i,j]
                    *(data_c + i * ldc + j) = acc * alpha + *(data_c + i * ldc + j) * beta;
                }
            }
        }
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

#ifdef OPENBLAS_AVAILABLE

/** @brief
 * BLAS-accelerated implementation of double-precision general matrix multiplication (DGEMM) in-place
 * Uses highly optimized OpenBLAS library for matrix multiplication operations
 * 
 ** @param order
 * Memory layout ordering (RowMajor or ColumnMajor)
 *
 ** @param transa
 * Transposition flag for matrix A (NoTrans or Trans)
 *
 ** @param transb  
 * Transposition flag for matrix B (NoTrans or Trans)
 *
 ** @param m
 * Number of rows in matrices A and C
 *
 ** @param n
 * Number of columns in matrices B and C
 *
 ** @param k
 * Number of columns in matrix A and rows in matrix B
 *
 ** @param alpha
 * Scalar multiplier for the matrix product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @param c
 * Pointer to tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * Requires OpenBLAS library to be available and linked
 * Significantly faster than naive implementation
 *
 ** @note
 * All input tensors must be of FLOAT64 type and properly allocated
 *
 ** @note
 * Performs the operation: C = alpha * op(A) * op(B) + beta * C
 * where op(X) is either X or X^T depending on transpose flags
 *
 ** @example
 * // Multiply matrices using BLAS: C = alpha * A * B + beta * C
 * blas_dgemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @see cblas_dgemm()
 **/
void blas_dgemminplace(const nnl2_order order, const nnl2_transpose transa, 
                       const nnl2_transpose transb, const int m, const int n, 
                       const int k, const double alpha, const nnl2_tensor* a, const int lda,
                       const nnl2_tensor* b, const int ldb, const double beta, nnl2_tensor* c,
                       const int ldc) {
						   
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif						

	// NNL2_INFO("Calling cblas_dgemm with order=%d, transa=%d, transb=%d, m=%d, n=%d, k=%d, alpha=%lf, a=%p, lda=%d, b=%p, ldb=%d, beta=%lf, c=%p, ldc=%d. a shape = [%d, %d], b shape = [%d, %d], c shape = [%d, %d]",
	// 		     order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, a -> shape [0], a -> shape [1], b -> shape [0], b -> shape [1], c -> shape [0], c -> shape [1]);
    
	// Cast data from void* to double*
	double* a_data = (double*)a->data;
	double* b_data = (double*)b->data;
	double* c_data = (double*)c->data;
	
	// Convert nnl2 order enum to CBLAS order enum
	CBLAS_ORDER cblas_order;
	
	switch(order) {
		case nnl2RowMajor:
			cblas_order = CblasRowMajor;
			break;
			
		case nnl2ColMajor:
			cblas_order = CblasColMajor;
			break;
			
		default: {
			NNL2_ORDER_ERROR(order);
			return;
		}
	}
	
	// Convert nnl2 transpose flags to CBLAS transpose enums
	CBLAS_TRANSPOSE cblas_transa;
	CBLAS_TRANSPOSE cblas_transb;
	
	switch(transa) {
		case nnl2NoTrans:
			cblas_transa = CblasNoTrans;  // Use matrix A as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transa = CblasTrans;    // Use transpose of matrix A
			break;
			
		default: {
			NNL2_TRANS_ERROR(transa);
			return;
		}
	}
	
	switch(transb) {
		case nnl2NoTrans:
			cblas_transb = CblasNoTrans;  // Use matrix B as-is (no transposition)
			break;
			
		case nnl2Trans:
			cblas_transb = CblasTrans;    // Use transpose of matrix B
			break;
			
		default: {
			NNL2_TRANS_ERROR(transb);
			return;
		}
	}
	
	// Call the actual BLAS DGEMM function for double-precision matrices
    // This is the highly optimized matrix multiplication routine from OpenBLAS
    // Performs: C = alpha * op(A) * op(B) + beta * C					   
	cblas_dgemm(cblas_order,     // Memory ordering (RowMajor/ColMajor)
                cblas_transa,    // Transpose flag for matrix A
                cblas_transb,    // Transpose flag for matrix B
                m,               // Number of rows in matrices A and C
                n,               // Number of columns in matrices B and C
                k,               // Number of columns in A and rows in B
                alpha,           // Scalar multiplier for A*B product
                a_data,          // Pointer to matrix A data
                lda,             // Leading dimension of matrix A
                b_data,          // Pointer to matrix B data
                ldb,             // Leading dimension of matrix B
                beta,            // Scalar multiplier for matrix C
                c_data,          // Pointer to matrix C data (output, modified in-place)
                ldc);            // Leading dimension of matrix C
				
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

#endif

/** @ingroup backend_system
 ** @brief Backend implementations for inplace_fill
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - naive_dgemminplace: Basic reference implementation
 *  - blas_dgemminplace: BLAS-accelerated implementation 
 *
 ** @see naive_dgemminplace
 ** @see blas_dgemminplace
 **/
nnl2_runtime_implementation dgemminplace_backends[] = {
	REGISTER_BACKEND(naive_dgemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
	
	#ifdef OPENBLAS_AVAILABLE
		REGISTER_BACKEND(blas_dgemminplace, nnl2_blas, BLAS_BACKEND_NAME),
	#endif
};

/**
 * @brief Function pointer for dgemm in-place
 * @ingroup backend_system 
 */
dgemminplacefn dgemminplace;

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(gemm);

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_dgemminplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(dgemminplace_backends, dgemminplace, backend_name, CURRENT_BACKEND(gemm));
}

/** 
 * @brief Gets the name of the active backend for inplace_all
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* get_gemm_backend() {
	return CURRENT_BACKEND(gemm);
}

/** 
 * @brief Function declaration for getting all `dgemminplace` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(dgemminplace);

/**
 * @brief Function declaration for getting the number of all `dgemminplace` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(dgemminplace);

#endif
