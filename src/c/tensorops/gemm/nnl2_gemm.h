#ifndef NNL2_GEMM_H
#define NNL2_GEMM_H

/** @brief
 * Type-agnostic general matrix multiplication with automatic output allocation
 * Automatically detects input data type and calls appropriate precision version
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
 * Scalar multiplier for the matrix nnl2_product A*B
 *
 ** @param a
 * Pointer to tensor containing matrix A (FLOAT32 or FLOAT64)
 *
 ** @param lda
 * Leading dimension of matrix A (stride between rows/columns)
 *
 ** @param b
 * Pointer to tensor containing matrix B (must match A's data type)
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
 * Automatically determines precision based on input tensor A data type
 * Supports both single (FLOAT32) and double (FLOAT64) precision
 *
 ** @note
 * Matrices A and B must have the same data type
 * Result tensor will have the same data type as input matrices
 *
 ** @example
 * // Multiply matrices with automatic type detection
 * nnl2_tensor* result = gemm(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, 
 *                      m, n, k, alpha, A, lda, B, ldb, beta);
 *
 ** @see sgemm()
 ** @see dgemm()
 **/
nnl2_tensor* gemm(const nnl2_order order, const nnl2_transpose transa, 
				  const nnl2_transpose transb, const int m, const int n, 
				  const int k, const double alpha, const nnl2_tensor* a, const int lda,
				  const nnl2_tensor* b, const int ldb, const double beta) {
		
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	
		
	// Determine data type from input tensor A			
	nnl2_tensor_type dtype = a->dtype;
	
	// Dispatch to appropriate precision implementation
	switch(dtype) {
		case FLOAT64: return dgemm(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta);
		case FLOAT32: return sgemm(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta);
		case INT64:   return i64gemm(order, transa, transb, m, n, k, (const int64_t)alpha, a, lda, b, ldb, (const int64_t)beta);
		case INT32:   return i32gemm(order, transa, transb, m, n, k, (const int32_t)alpha, a, lda, b, ldb, (const int32_t)beta);	
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

#endif
