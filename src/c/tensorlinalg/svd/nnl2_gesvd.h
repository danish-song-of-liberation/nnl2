#ifndef NNL2_GESVD_H
#define NNL2_GESVD_H

// NNL2

int nnl2_gesvd(const nnl2_order order, const char jobu, const char jobvt,
               const int m, const int n, nnl2_tensor* a, const int lda,
               nnl2_tensor* s, nnl2_tensor* u, const int ldu,
               nnl2_tensor* vt, const int ldvt, nnl2_tensor* superb) {
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    switch(a->dtype) {
        case FLOAT64:  return nnl2_f64dgesvd(order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
        case FLOAT32:  return nnl2_f32sgesvd(order, jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt, superb);
        
        default: {
            NNL2_TYPE_ERROR(a->dtype);
            return -1;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_GESVD_H **/
