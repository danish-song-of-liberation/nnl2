#ifndef NNL2_ASSIGN_ROW_ADD_H
#define NNL2_ASSIGN_ROW_ADD_H

// NNL2

void nnl2_naive_assign_row_add(nnl2_tensor* dst, int seq_index, nnl2_tensor* src) {
    size_t elem_size = get_dtype_size(src->dtype);

    for(int b = 0; b < src->shape[0]; b++) {
        char* src_ptr = (char*)src->data;
        src_ptr += b * src->strides[0] * elem_size;
        src_ptr += seq_index * src->strides[1] * elem_size;

        char* dst_ptr = (char*)dst->data;
        dst_ptr += b * dst->strides[0] * elem_size;

        int F = src->shape[2];

        switch(src->dtype) {
			case FLOAT64: {
                nnl2_float64* s = (nnl2_float64*)src_ptr;
                nnl2_float64* d = (nnl2_float64*)dst_ptr;
                for(int f = 0; f < F; ++f) d[f] += s[f];
                break;
            }
			
            case FLOAT32: {
                nnl2_float32* s = (nnl2_float32*)src_ptr;
                nnl2_float32* d = (nnl2_float32*)dst_ptr;
                for(int f = 0; f < F; ++f) d[f] += s[f];
                break;
            }
			
			case INT32: {
                nnl2_int32* s = (nnl2_int32*)src_ptr;
                nnl2_int32* d = (nnl2_int32*)dst_ptr;
                for(int f = 0; f < F; ++f) d[f] += s[f];
                break;
            }

            default: {
                NNL2_TYPE_ERROR(src->dtype);
				break;
			}
        }
    }
	
	return;
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for assign_row_add operation
 * @details
 * assign_row_add is the backward (accumulating) counterpart of assign_row.
 * It gathers a row from a 3D tensor and accumulates it into a 2D tensor.
 *
 * Currently registered backends:
 *  - nnl2_naive: reference implementation
 */
Implementation assign_row_add_backends[] = {
    REGISTER_BACKEND(nnl2_naive_assign_row_add, nnl2_naive, NAIVE_BACKEND_NAME),
};

/**
 * @brief Function pointer for assign_row_add operation
 * @ingroup backend_system
 */
assignrowfn nnl2_assign_row_add;

/**
 * @brief Sets the backend for assign_row_add operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 */
void set_assign_row_add_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(assign_row_add_backends, nnl2_assign_row_add, backend_name);
}

#endif /** NNL2_ASSIGN_ROW_ADD_H **/


