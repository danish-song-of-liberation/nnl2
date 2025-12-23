#ifndef NNL2_GESDD_H
#define NNL2_GESDD_H

// NNL2

int nnl2_gesdd(const nnl2_order order, const char jobz,
               const int m, const int n, nnl2_tensor* a, const int lda,
               nnl2_tensor* s, nnl2_tensor* u, const int ldu,
               nnl2_tensor* vt, const int ldvt, nnl2_tensor* iwork) {
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    switch(a->dtype) {
        case FLOAT64:  return nnl2_f64dgesdd(order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, iwork);
        case FLOAT32:  return nnl2_f32sgesdd(order, jobz, m, n, a, lda, s, u, ldu, vt, ldvt, iwork);
        
        default: {
            NNL2_TYPE_ERROR(a->dtype);
            return -1;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_GESDD_H **/
