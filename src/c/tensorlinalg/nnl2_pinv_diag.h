#ifndef NNL2_PINV_DIAG_H
#define NNL2_PINV_DIAG_H

#include <math.h>

/** @brief
 * Computes safe reciprocal of singular values for pseudoinverse
 *
 ** @param in
 * Pointer to input 1D tensor (singular values)
 *
 ** @param eps
 * Threshold for inversion stability
 *
 ** @return nnl2_tensor*
 * Pointer to a new tensor containing inverted singular values
 *
 ** @see nnl2_diag_vector_matrix
 ** @see nnl2_pinv
 **/
nnl2_tensor* nnl2_pinv_diag(nnl2_tensor* in, double eps) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(in == NULL) {
            NNL2_ERROR("In function nnl2_pinv_diag, passed tensor is NULL. returning NULL");
            return NULL;
        }

        if(in->rank != 1) {
            NNL2_ERROR("In function nnl2_pinv_diag, input tensor must be 1D");
            return NULL;
        }
    #endif

    nnl2_tensor* out = nnl2_zeros_like(in);

    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(out == NULL) {
            NNL2_ERROR("In function nnl2_pinv_diag, allocated out is NULL. returning NULL");
            return NULL;
        }
    #endif

    size_t n = in->shape[0];

    switch(out->dtype) {
        case FLOAT64: {
            nnl2_float64* src = (nnl2_float64*)in->data;
            nnl2_float64* dst = (nnl2_float64*)out->data;

            for(size_t i = 0; i < n; ++i) {
                nnl2_float64 s = src[i];
                dst[i] = (fabs(s) > eps) ? (1.0 / s) : 0.0;
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* src = (nnl2_float32*)in->data;
            nnl2_float32* dst = (nnl2_float32*)out->data;

            for(size_t i = 0; i < n; ++i) {
                nnl2_float32 s = src[i];
                dst[i] = (fabsf(s) > (nnl2_float32)eps) ? (1.0f / s) : 0.0f;
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(out->dtype);
            return NULL;
        }
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return out;
}

#endif /** NNL2_PINV_DIAG_H **/
