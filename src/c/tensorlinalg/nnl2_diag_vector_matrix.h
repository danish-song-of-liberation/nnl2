#ifndef NNL2_DIAG_VECTOR_MATRIX_H
#define NNL2_DIAG_VECTOR_MATRIX_H

/** @brief 
 * Create a diagonal matrix from a vector
 *
 ** @param vector 
 * Input 1D tensor
 *
 ** @param k 
 * The diagonal index:
 * k = 0: main diagonal
 * k > 0: diagonal above the main
 * k < 0: diagonal below the main
 *
 ** @return nnl2_tensor* 
 * A new 2D tensor with the vector on the specified diagonal
 */
nnl2_tensor* nnl2_diag_vector_matrix(nnl2_tensor* vector, int k) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(vector == NULL) {
			NNL2_ERROR("In function nnl2_diag_vector_matrix, passed vector is NULL. returning NULL");
			return NULL;
		}
		
		if(vector -> rank != 1) {
			NNL2_ERROR("In function nnl2_diag_vector_matrix, expected 1D vector, got: %d", vector -> rank);
			return NULL;
		}
	#endif
	
	size_t len = vector -> shape[0];
    size_t size = len + (k > 0 ? k : -k);  
	size_t rows = size;
	size_t cols = size;
	
	nnl2_tensor* out = nnl2_zeros((int32_t[]){ rows, cols }, 2, vector -> dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(out == NULL) {
			NNL2_TENSOR_ERROR("zeros");
			return NULL;
		}
	#endif
	
	switch(out -> dtype) {
		case FLOAT64: {
			nnl2_float64* out_data = (nnl2_float64*)out->data;
			nnl2_float64* vec_data = (nnl2_float64*)vector->data;

			// Fill the diagonal with vector elements
			for(size_t i = 0; i < len; i++) {
				size_t r = i;
				size_t c = i + k;
				if(r < rows && c < cols) {
					out_data[r * cols + c] = vec_data[i];
				}
			}
			
			break;
		}
		
		case FLOAT32: {
			nnl2_float32* out_data = (nnl2_float32*)out->data;
			nnl2_float32* vec_data = (nnl2_float32*)vector->data;

			for(size_t i = 0; i < len; i++) {
				size_t r = i;
				size_t c = i + k;
				if(r < rows && c < cols) {
					out_data[r * cols + c] = vec_data[i];
				}
			}
			
			break;
		}
		
		case INT64: {
			nnl2_int64* out_data = (nnl2_int64*)out->data;
			nnl2_int64* vec_data = (nnl2_int64*)vector->data;

			for(size_t i = 0; i < len; i++) {
				size_t r = i;
				size_t c = i + k;
				if(r < rows && c < cols) {
					out_data[r * cols + c] = vec_data[i];
				}
			}
			
			break;
		}
		
		case INT32: {
			nnl2_int32* out_data = (nnl2_int32*)out->data;
			nnl2_int32* vec_data = (nnl2_int32*)vector->data;

			for(size_t i = 0; i < len; i++) {
				size_t r = i;
				size_t c = i + k;
				if(r < rows && c < cols) {
					out_data[r * cols + c] = vec_data[i];
				}
			}
			
			break;
		}
		
		default: {
			// Unsupported dtype
			NNL2_TYPE_ERROR(out -> dtype);
			return NULL;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif 
	
	return out;
}

#endif /** NNL2_DIAG_VECTOR_MATRIX_H **/
