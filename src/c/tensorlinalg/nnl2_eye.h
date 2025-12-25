#ifndef NNL2_EYE_H
#define NNL2_EYE_H

/** @brief
 * Creates an identity matrix tensor
 * 
 ** @param rows 
 * Number of rows in the output tensor
 *
 ** @param cols 
 * Number of columns in the output tensor
 *
 ** @param dtype 
 * Data type of the tensor elements
 * 
 ** @return nnl2_tensor*
 * Pointer to the created identity matrix tensor
 * 
 ** @warning 
 * The caller is responsible for freeing the returned tensor using
 * nnl2_free_tensor()
 */
nnl2_tensor* nnl2_eye(size_t rows, size_t cols, nnl2_tensor_type dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
    nnl2_tensor* t = nnl2_zeros((int[]){ rows, cols }, 2, dtype);
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(t == NULL) {
			NNL2_TENSOR_ERROR("zeros");
			return NULL;
		}
	#endif

    size_t n = rows < cols ? rows : cols;
    size_t stride = t->strides[0];

    switch (dtype) {
		case FLOAT64: {
			nnl2_float64* data = (nnl2_float64*)t->data;
			for(size_t i = 0; i < n; ++i)  data[i * stride + i] = 1.0;
			break;
		}
		
		case FLOAT32: {
			nnl2_float32* data = (nnl2_float32*)t->data;
			for(size_t i = 0; i < n; ++i)  data[i * stride + i] = 1.0f;
			break;
		}
		
		case INT32: {
			nnl2_int32* data = (nnl2_int32*)t->data;
			for(size_t i = 0; i < n; ++i)  data[i * stride + i] = 1.0;
			break;
		}
		
		case INT64: {
			nnl2_int64* data = (nnl2_int64*)t->data;
			for(size_t i = 0; i < n; ++i)  data[i * stride + i] = 1.0;
			break;
		}
		
		default: {
			nnl2_free_tensor(t);
			return NULL;
		}
    }

	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif

    return t;
}

#endif /** NNL2_EYE_H **/
