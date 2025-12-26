#ifndef NNL2_TRIU_H
#define NNL2_TRIU_H

/** @brief 
 * Returns the upper triangular part of a 2D tensor
 *
 ** @param in 
 * Pointer to the input 2D tensor (matrix)
 *
 ** @param k 
 * Diagonal offset
 *
 ** @return nnl2_tensor* 
 * Pointer to a new tensor containing the upper triangular part
 *
 ** @see nnl2_tril
 ** @see nnl2_zeros_like
 ** @see nnl2_free_tensor
 **/
nnl2_tensor* nnl2_triu(nnl2_tensor* in, int k) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(in == NULL) {
			NNL2_ERROR("In function nnl2_triu, passed tensor is NULL. returning NULL");
			return NULL;
		}
	#endif
	
	nnl2_tensor* out = nnl2_zeros_like(in);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(out == NULL) {
			NNL2_ERROR("In function nnl2_triu, allocated out is NULL. returning NULL");
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
				int j0 = (int)i + k;
				if(j0 < 0) j0 = 0;
				if(j0 > (int)cols) j0 = cols;

				for(int j = 0; j < j0; ++j)
					dst[i*cols + j] = 0.0;

				for(size_t j = j0; j < cols; ++j)
					dst[i*cols + j] = src[i*cols + j];
			}
			
			break;
		}
		
		case FLOAT32: {
			nnl2_float32* dst = (nnl2_float32*)out->data;
			nnl2_float32* src = (nnl2_float32*)in->data;

			for(size_t i = 0; i < rows; ++i) {
				int j0 = (int)i + k;
				if(j0 < 0) j0 = 0;
				if(j0 > (int)cols) j0 = cols;

				for(int j = 0; j < j0; ++j)
					dst[i*cols + j] = 0.0f;

				for(size_t j = j0; j < cols; ++j)
					dst[i*cols + j] = src[i*cols + j];
			}
			
			break;
		}
		
		case INT64: {
			nnl2_int64* dst = (nnl2_int64*)out->data;
			nnl2_int64* src = (nnl2_int64*)in->data;

			for(size_t i = 0; i < rows; ++i) {
				int j0 = (int)i + k;
				if(j0 < 0) j0 = 0;
				if(j0 > (int)cols) j0 = cols;

				for(int j = 0; j < j0; ++j)
					dst[i*cols + j] = 0;

				for(size_t j = j0; j < cols; ++j)
					dst[i*cols + j] = src[i*cols + j];
			}
			
			break;
		}
		
		case INT32: {
			nnl2_int32* dst = (nnl2_int32*)out->data;
			nnl2_int32* src = (nnl2_int32*)in->data;

			for(size_t i = 0; i < rows; ++i) {
				int j0 = (int)i + k;
				if(j0 < 0) j0 = 0;
				if(j0 > (int)cols) j0 = cols;

				for(int j = 0; j < j0; ++j)
					dst[i*cols + j] = 0;

				for(size_t j = j0; j < cols; ++j)
					dst[i*cols + j] = src[i*cols + j];
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

#endif /** NNL2_TRIU_H **/
