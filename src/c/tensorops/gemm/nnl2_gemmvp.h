#ifndef NNL2_GEMMVP_H
#define NNL2_GEMMVP_H

nnl2_tensor* nnl2_gemmvp(nnl2_tensor* a, nnl2_tensor* b, nnl2_tensor* vector) {
	nnl2_tensor* c = nnl2_quick_gemm(a, b);
	add_broadcasting_inplace(c, vector);
	
	return c;
}	

#endif /** NNL2_GEMMVP_H **/
