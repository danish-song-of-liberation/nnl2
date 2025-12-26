#ifndef NNL2_I64GEMM_H
#define NNL2_I64GEMM_H

/** @brief
 * 64-bit integer general matrix multiplication with automatic output allocation
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
 * Pointer to tensor containing matrix A (must be INT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must be INT64)
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
 * Automatically allocates output tensor
 * Result must be freed using nnl2_free_tensor() to avoid memory leaks
 *
 ** @note
 * Performs: C = alpha * op(A) * op(B) + beta * C
 * where C is allocated and filled before operation
 *
 ** @example
 * // Multiply matrices and get new result tensor
 * nnl2_tensor* result = i64gemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, 
 *                               m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see i64gemminplace()
 **/
nnl2_tensor* i64gemm(const nnl2_order order, const nnl2_transpose transa, 
		        const nnl2_transpose transb, const int m, const int n, 
			    const int k, const int64_t alpha, const nnl2_tensor* a, const int lda,
			    const nnl2_tensor* b, const int ldb, const int64_t beta) {
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	
	
	// Define shape and properties for result matrix C
	int shape_c[] = {m, n}; // Result matrix dimensions: m x n
	int rank_c = 2; 		// 2D matrix
	nnl2_tensor_type type_c = INT64;
	
	// Create output tensor
	nnl2_tensor* c = nnl2_empty(shape_c, rank_c, type_c);
	
	if (!c) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_BASIC
			NNL2_ERROR("Failed to allocate output tensor for i64gemm");
		#endif
		return NULL;
	}
	
	// Perform in-place matrix multiplication on the created tensor
	i64gemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, n);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
	
	return c; // Return the newly created result tensor
}

#endif /** NNL2_I64GEMM_H **/
