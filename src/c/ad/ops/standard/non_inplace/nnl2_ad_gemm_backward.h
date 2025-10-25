#ifndef NNL2_AD_GEMM_BACKWARD_H
#define NNL2_AD_GEMM_BACKWARD_H

void nnl2_ad_reverse_derivative_gemm(nnl2_ad_tensor* out_tensor, nnl2_ad_tensor* addend, nnl2_ad_tensor* sumend) { 
    int m = out_tensor->grad->shape[0];
    int k = out_tensor->grad->shape[1];
    int n = sumend->data->shape[1];
    
    nnl2_tensor* grad_out_a = gemm(
        nnl2RowMajor, 		     // order
        nnl2NoTrans,    		 // transa
        nnl2Trans,          	 // transb
        m, 					     // rows of result
        addend->data->shape[1],  // cols of result  
        k, 						 // common dimension
        1.0, 					 // alpha
        out_tensor->grad,        // A
        k, 						 // lda
        sumend->data,		     // B
        n,  					 // ldb
        0.0  					 // beta
    );
    
    nnl2_tensor* grad_out_b = gemm(
        nnl2RowMajor,   		 // order
        nnl2Trans,  	    	 // transa
        nnl2NoTrans, 		     // transb
        addend->data->shape[0],  // rows of result
        n,  					 // cols of result
        m,  					 // common dimension
        1.0, 					 // alpha
        addend->data, 			 // A
        k,  					 // lda  
        out_tensor->grad, 		 // B
        n,  					 // ldb
        0.0  					 // beta
    );
    
    if(addend->requires_grad) nnl2_add_inplace(addend->grad, grad_out_a);
    if(sumend->requires_grad) nnl2_add_inplace(sumend->grad, grad_out_b);
    nnl2_free_tensor(grad_out_a);
    nnl2_free_tensor(grad_out_b);
}

#endif /** NNL2_AD_GEMM_BACKWARD_H **/
