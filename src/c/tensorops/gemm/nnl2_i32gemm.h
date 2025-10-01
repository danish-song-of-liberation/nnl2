#ifndef NNL2_I32GEMM_H
#define NNL2_I32GEMM_H

/** @brief
 * Double-precision general matrix multiplication with automatic output allocation
 * Creates a new tensor for the result and performs matrix multiplication
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
 * Pointer to tensor containing matrix A (must be FLOAT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must be FLOAT64)
 *
 ** @param ldb
 * Leading dimension of matrix B (stride between rows/columns)
 *
 ** @param beta
 * Scalar multiplier for matrix C (before addition)
 *
 ** @return
 * Pointer to newly allocated tensor containing result matrix C, or NULL on failure
 *
 ** @note
 * Automatically allocates output tensor filled with ones
 * Result must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * Performs: C = alpha * op(A) * op(B) + beta * C
 * where C is initialized with ones
 *
 ** @example
 * // Multiply matrices and get new result tensor
 * Tensor* result = dgemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see dgemminplace()
 **/
Tensor* i32gemm(const nnl2_order order, const nnl2_transpose transa, 
		        const nnl2_transpose transb, const int m, const int n, 
			    const int k, const int32_t alpha, const Tensor* a, const int lda,
			    const Tensor* b, const int ldb, const int32_t beta) {
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	
	
	// Define shape and properties for result matrix C
	int shape_c[] = {m, n}; // Result matrix dimensions: m x n
	int rank_c = 2; 		// 2D matrix
	TensorType type_c = INT32;
	
	// Create output tensor
	Tensor* c = nnl2_empty(shape_c, rank_c, type_c);
	
	// Perform in-place matrix multiplication on the created tensor
	i32gemminplace(order, transa, transb, m, n, k, (const int32_t)alpha, a, lda, b, ldb, (const int32_t)beta, c, n);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
	
	return c; // Return the newly created result tensor
}

#endif /** NNL2_I32GEMM_H **/
