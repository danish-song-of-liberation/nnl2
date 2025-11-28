#ifndef NNL2_QUICK_GEMM_H
#define NNL2_QUICK_GEMM_H

nnl2_tensor* nnl2_quick_gemm(nnl2_tensor* a, nnl2_tensor* b) {
	nnl2_order order = nnl2RowMajor;
	
	nnl2_transpose transa = nnl2NoTrans;
	nnl2_transpose transb = nnl2NoTrans;
	
	int m = a->shape[0];
    int k = a->shape[1];
    int n = b->shape[1];
	
	int lda = k;
    int ldb = n;
	
	double alpha = 1.0;
    double beta  = 0.0;

    return gemm(order, transa, transb, m, n, k, alpha, (nnl2_tensor*)a, lda, (nnl2_tensor*)b, ldb, beta);
}

#endif /** NNL2_QUICK_GEMM_H **/
