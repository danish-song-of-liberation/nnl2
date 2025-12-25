#ifndef NNL2_DIAG_MATRIX_VECTOR_H
#define NNL2_DIAG_MATRIX_VECTOR_H

/** @brief 
 * Extract a diagonal from a matrix and return it as a vector
 *
 ** @param matrix 
 * Input 2D tensor (matrix)
 *
 ** @param k 
 * The diagonal index:
 * k = 0: main diagonal
 * k > 0: diagonal above the main
 * k < 0: diagonal below the main
 *
 ** @return nnl2_tensor* 
 * A new 1D tensor containing the specified diagonal
 */
nnl2_tensor* nnl2_diag_matrix_vector(nnl2_tensor* matrix, int k) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif 
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(matrix == NULL) {
            NNL2_ERROR("In function nnl2_diag_matrix_vector, passed matrix is NULL. returning NULL");
            return NULL;
        }
        
        if(matrix -> rank != 2) {
            NNL2_ERROR("In function nnl2_diag_matrix_vector, expected 2D matrix, got: %d", matrix -> rank);
            return NULL;
        }
    #endif
    
    size_t rows = matrix -> shape[0];
    size_t cols = matrix -> shape[1];
    
    size_t diag_len;
    if (k >= 0) {
        // For k >= 0 diagonal starts at (0, k) goes to (min(rows, cols-k)-1, min(cols-k, rows)+k-1)
        diag_len = (rows < cols - k) ? rows : cols - k;
    } else {
        // For k < 0 diagonal starts at (-k, 0) goes to (min(rows+k, cols)-1, min(cols, rows+k)-k-1)
        diag_len = (cols < rows + k) ? cols : rows + k;
    }
    
    if (k >= 0 && k >= (int)cols) diag_len = 0;
    if (k < 0 && -k >= (int)rows) diag_len = 0;
    
    nnl2_tensor* out = nnl2_zeros((int32_t[]){ (int32_t)diag_len }, 1, matrix -> dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (out == NULL) {
			NNL2_ERROR("In function nnl2_diag_matrix_vector, failed to allocate output vector");
			return NULL;
		}
	#endif
    
    if(diag_len == 0) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_FUNC_EXIT();
        #endif 
		
        return out;
    }
    
    switch(matrix -> dtype) {
        case FLOAT64: {
            nnl2_float64* mat_data = (nnl2_float64*)matrix->data;
            nnl2_float64* out_data = (nnl2_float64*)out->data;
            
            // Extract diagonal elements
            for(size_t i = 0; i < diag_len; i++) {
                size_t r = (k >= 0) ? i : i - k;
                size_t c = (k >= 0) ? i + k : i;
                
                if (r < rows && c < cols) {
                    out_data[i] = mat_data[r * cols + c];
                }
            }
			
            break;
        }
        
        case FLOAT32: {
            nnl2_float32* mat_data = (nnl2_float32*)matrix->data;
            nnl2_float32* out_data = (nnl2_float32*)out->data;
            
            for(size_t i = 0; i < diag_len; i++) {
                size_t r = (k >= 0) ? i : i - k;
                size_t c = (k >= 0) ? i + k : i;
                
                if (r < rows && c < cols) {
                    out_data[i] = mat_data[r * cols + c];
                }
            }
			
            break;
        }
        
        case INT64: {
            nnl2_int64* mat_data = (nnl2_int64*)matrix->data;
            nnl2_int64* out_data = (nnl2_int64*)out->data;
            
            for(size_t i = 0; i < diag_len; i++) {
                size_t r = (k >= 0) ? i : i - k;
                size_t c = (k >= 0) ? i + k : i;
                
                if (r < rows && c < cols) {
                    out_data[i] = mat_data[r * cols + c];
                }
            }
			
            break;
        }
        
        case INT32: {
            nnl2_int32* mat_data = (nnl2_int32*)matrix->data;
            nnl2_int32* out_data = (nnl2_int32*)out->data;
            
            for(size_t i = 0; i < diag_len; i++) {
                size_t r = (k >= 0) ? i : i - k;
                size_t c = (k >= 0) ? i + k : i;
                
                if (r < rows && c < cols) {
                    out_data[i] = mat_data[r * cols + c];
                }
            }
			
            break;
        }
        
        default: {
            // Unsupported dtype
            NNL2_TYPE_ERROR(matrix -> dtype);
            nnl2_free_tensor(out);
            return NULL;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif 
    
    return out;
}

#endif /** NNL2_DIAG_MATRIX_VECTOR_H **/
