#ifndef NNL2_SWAP_ROWS_H
#define NNL2_SWAP_ROWS_H

/** @brief 
 * Swaps two rows in a 2D tensor (matrix)
 * 
 ** @param out 
 * Pointer to the tensor whose rows will be swapped
 * 
 ** @param row1 
 * Zero-based index of the first row to swap
 *
 ** @param row2 
 * Zero-based index of the second row to swap
 */
void nnl2_swap_rows(nnl2_tensor* out, size_t row1, size_t row2) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif 
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(out == NULL) {
			NNL2_ERROR("In function nnl2_tensor_swap_rows, passed tensor is NULL. nothing to swap");
			return;
		}
	#endif
	
    size_t cols = out->shape[1];

    switch(out->dtype) {
        case FLOAT64: {
            nnl2_float64* data = (nnl2_float64*)out->data;
            for(size_t j = 0; j < cols; j++) {
                nnl2_float64 tmp = data[row1 * cols + j];
                data[row1 * cols + j] = data[row2 * cols + j];
                data[row2 * cols + j] = tmp;
            }
			
            break;
        }

        case FLOAT32: {
            nnl2_float32* data = (nnl2_float32*)out->data;
            for(size_t j = 0; j < cols; j++) {
                nnl2_float32 tmp = data[row1 * cols + j];
                data[row1 * cols + j] = data[row2 * cols + j];
                data[row2 * cols + j] = tmp;
            }
			
            break;
        }

        case INT64: {
            nnl2_int64* data = (nnl2_int64*)out->data;
            for(size_t j = 0; j < cols; j++) {
                nnl2_int64 tmp = data[row1 * cols + j];
                data[row1 * cols + j] = data[row2 * cols + j];
                data[row2 * cols + j] = tmp;
            }
			
            break;
        }

        case INT32: {
            nnl2_int32* data = (nnl2_int32*)out->data;
            for(size_t j = 0; j < cols; j++) {
                nnl2_int32 tmp = data[row1 * cols + j];
                data[row1 * cols + j] = data[row2 * cols + j];
                data[row2 * cols + j] = tmp;
            }
			
            break;
        }

        default: {
            NNL2_TYPE_ERROR(out->dtype);
            return;
		}
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return;
}

#endif /** NNL2_SWAP_ROWS_H **/
