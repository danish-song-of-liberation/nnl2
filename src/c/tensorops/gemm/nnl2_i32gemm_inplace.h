#ifndef NNL2_I32GEMM_INPLACE_H
#define NNL2_I32GEMM_INPLACE_H

/** @brief
 * Naive implementation of int32 general matrix multiplication (i32gemm) in-place
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
 * All input tensors must be of INT32 type and properly allocated
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
 * naive_i32gemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, 
 *                   m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @exception
 * Function returns early with error message if invalid parameters are detected
 *
 **/
void naive_i32gemminplace(const nnl2_order order, const nnl2_transpose transa, 
                         const nnl2_transpose transb, const int m, const int n, 
                         const int k, const int32_t alpha, const nnl2_tensor* a, const int lda,
                         const nnl2_tensor* b, const int ldb, const int32_t beta, nnl2_tensor* c,
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
			NNL2_ERROR("lda is less than number of columns of a matrix (gemm)");
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
	
	// Casting tensor data to int32_t with volatile to prevent compiler optimizations
    volatile int32_t* data_a = (volatile int32_t*)a->data;
    volatile int32_t* data_b = (volatile int32_t*)b->data;
    volatile int32_t* data_c = (volatile int32_t*)c->data;                          
    
    if(order == nnl2RowMajor) {
		// nnl2_runtime_implementation for RowMajor order (lowercase data organization)
		
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {    
                int32_t acc = 0;
        
                for(volatile int l = 0; l < k; l++) {
                    int32_t a_val; 
                    int32_t b_val; 
                
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
		// nnl2_runtime_implementation for ColumnMajor order (column-based data organization)
		
        for(volatile int i = 0; i < m; i++) {
            for(volatile int j = 0; j < n; j++) {
                int32_t acc = 0;
                
                for(volatile int l = 0; l < k; l++) {
                    int32_t a_val;
                    int32_t b_val;
                
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

/** @ingroup backend_system
 ** @brief Backend implementations for i32gemminplace
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - naive_i32gemminplace: A simple, unremarkable naive implementation of matrix multiplication
 *
 ** @see naive_i32gemminplace
 **/
nnl2_runtime_implementation i32gemminplace_backends[] = {	
	REGISTER_BACKEND(naive_i32gemminplace, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for i32gemm in-place
 * @ingroup backend_system 
 */
i32gemminplacefn i32gemminplace;

/** 
 * @brief Sets the backend for view
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void set_i32gemminplace_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(i32gemminplace_backends, i32gemminplace, backend_name);
}

#endif /** NNL2_I32GEMM_INPLACE_H **/
