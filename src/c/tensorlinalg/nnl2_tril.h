#ifndef NNL2_TRIL_H
#define NNL2_TRIL_H

/** @brief 
 * Returns the lower triangular part of a 2D tensor
 *
 ** @param in 
 * Pointer to the input 2D tensor (matrix)
 *
 ** @param k 
 * Diagonal offset
 *
 ** @return nnl2_tensor* 
 * Pointer to a new tensor containing the lower triangular part
 *
 ** @see nnl2_triu
 ** @see nnl2_zeros_like
 ** @see nnl2_free_tensor
 **/
nnl2_tensor* nnl2_tril(nnl2_tensor* in, int k) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif 
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(in == NULL) {
            NNL2_ERROR("In function nnl2_tril, passed tensor is NULL. returning NULL");
            return NULL;
        }
    #endif
    
    nnl2_tensor* out = nnl2_zeros_like(in);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(out == NULL) {
            NNL2_ERROR("In function nnl2_tril, allocated out is NULL. returning NULL");
            return NULL;
        }
    #endif
    
    size_t rows = in->shape[0];
    size_t cols = in->shape[1];

    switch(out->dtype) {
        case FLOAT64: {
            nnl2_float64* dst = (nnl2_float64*)out->data;
            nnl2_float64* src = (nnl2_float64*)in->data;

            for(size_t i = 0; i < rows; ++i) {
                int j1 = (int)i + k + 1;
                if(j1 < 0) j1 = 0;
                if(j1 > (int)cols) j1 = cols;

                for(size_t j = 0; j < (size_t)j1; ++j)
                    dst[i*cols + j] = src[i*cols + j];

                for(size_t j = j1; j < cols; ++j)
                    dst[i*cols + j] = 0.0;
            }
            
            break;
        }

        case FLOAT32: {
            nnl2_float32* dst = (nnl2_float32*)out->data;
            nnl2_float32* src = (nnl2_float32*)in->data;

            for(size_t i = 0; i < rows; ++i) {
                int j1 = (int)i + k + 1;
                if(j1 < 0) j1 = 0;
                if(j1 > (int)cols) j1 = cols;

                for(size_t j = 0; j < (size_t)j1; ++j)
                    dst[i*cols + j] = src[i*cols + j];

                for(size_t j = j1; j < cols; ++j)
                    dst[i*cols + j] = 0.0f;
            }
            
            break;
        }

        case INT64: {
            nnl2_int64* dst = (nnl2_int64*)out->data;
            nnl2_int64* src = (nnl2_int64*)in->data;

            for(size_t i = 0; i < rows; ++i) {
                int j1 = (int)i + k + 1;
                if(j1 < 0) j1 = 0;
                if(j1 > (int)cols) j1 = cols;

                for(size_t j = 0; j < (size_t)j1; ++j)
                    dst[i*cols + j] = src[i*cols + j];

                for(size_t j = j1; j < cols; ++j)
                    dst[i*cols + j] = 0;
            }
            
            break;
        }

        case INT32: {
            nnl2_int32* dst = (nnl2_int32*)out->data;
            nnl2_int32* src = (nnl2_int32*)in->data;

            for(size_t i = 0; i < rows; ++i) {
                int j1 = (int)i + k + 1;
                if(j1 < 0) j1 = 0;
                if(j1 > (int)cols) j1 = cols;

                for(size_t j = 0; j < (size_t)j1; ++j)
                    dst[i*cols + j] = src[i*cols + j];

                for(size_t j = j1; j < cols; ++j)
                    dst[i*cols + j] = 0;
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

#endif /** NNL2_TRIL_H **/
