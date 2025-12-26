#ifndef NNL2_GEMM_INPLACE_H
#define NNL2_GEMM_INPLACE_H

/** @brief
 * Type-agnostic in-place general matrix multiplication
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
 ** @param c
 * Pointer to output tensor for storing result matrix C (modified in-place)
 *
 ** @param ldc
 * Leading dimension of matrix C (stride between rows/columns)
 *
 ** @return
 * None (result is stored in-place in matrix C)
 *
 ** @note
 * Automatically determines precision based on input tensor A data type
 * Supports both single (FLOAT32) and double (FLOAT64) precision
 *
 ** @note
 * All input tensors must have the same data type
 * Matrix C is modified in-place and must be properly allocated
 *
 ** @example
 * // In-place matrix multiplication with automatic type detection
 * gemminplace(nnl2RowMajor, nnl2NoTrans, nnl2NoTrans, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
 *
 ** @see sgemminplace(), dgemminplace()
 **/
void gemminplace(const nnl2_order order, const nnl2_transpose transa, 
				 const nnl2_transpose transb, const int m, const int n, 
				 const int k, const double alpha, const nnl2_tensor* a, const int lda,
				 const nnl2_tensor* b, const int ldb, const double beta,
				 nnl2_tensor* c, const int ldc) {

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif	

	// Determine data type from input tensor A
	nnl2_tensor_type dtype = a->dtype;
	
	// Dispatch to appropriate precision implementation
	switch(dtype) {
		case FLOAT64: dgemminplace(order, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);						    	   break;
		case FLOAT32: sgemminplace(order, transa, transb, m, n, k, (const float)alpha, a, lda, b, ldb, (const float)beta, c, ldc);         break;
		case INT64:   i64gemminplace(order, transa, transb, m, n, k, (const int64_t)alpha, a, lda, b, ldb, (const int32_t)beta, c, ldc);   break;
		case INT32:   i32gemminplace(order, transa, transb, m, n, k, (const int32_t)alpha, a, lda, b, ldb, (const int32_t)beta, c, ldc);   break;
		
		default: {
			NNL2_TYPE_ERROR(dtype);
			return;
		}
	}			
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif	
}

#endif
