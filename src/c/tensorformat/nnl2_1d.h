#ifndef NNL2_1D_H
#define NNL2_1D_H

/** @brief 
 * Prints the contents of a 1D tensor to standard output
 *
 ** @param tensor
 * Pointer to the 1D tensor to be printed 
 *
 ** @param full_print
 * Flag controlling output truncation:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors (show first and last elements)
 *
 ** @param max_rows
 * Maximum number of rows to display without truncation
 *
 ** @param show_rows
 * Number of elements to show from beginning and end when truncating
 *
 ** @note
 * In safety mode, performs extensive validation of input parameters
 *
 * @note
 * Output format includes tensor metadata (type, shape) and formatted data
 *
 ** @example
 * // Print full tensor contents
 * nnl2_print_1d_tensor(my_tensor, true, 10, 3);
 *
 * // Print truncated version for large tensors  
 * nnl2_print_1d_tensor(large_tensor, false, 10, 3);
 */
void nnl2_print_1d_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t show_rows) {		
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif	
	
	int rows = tensor->shape[0];

	// Comprehensive input validation in safety mode
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (tensor == NULL) {
            NNL2_ERROR("NULL tensor pointer");
            return;
        }
    
        if (tensor->data == NULL) {
            NNL2_ERROR("Tensor data is NULL");
            return;
        }
        
        if (tensor->shape == NULL) {
            NNL2_ERROR("Tensor shape is NULL");
            return;
        }
        
        if (tensor->rank != 1) {  
            NNL2_ERROR("Expected 1D tensor, got %dD", tensor->rank);
            return;
        }
		
        if (rows <= 0) {
            NNL2_ERROR("Invalid tensor shape [%d]", rows);
            return;
        }
    #endif
	
	if(!full_print) {
		if(2 * show_rows >= max_rows) {
			NNL2_ERROR("show_rows (%d) Is too large for max_rows (%d). Check the correctness of the tensor formatting settings you have specified", show_rows, max_rows);
			return;
		}
	}
    
	// Get data type information for formatting	
    TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
    
    // Print tensor header with metadata
    printf("#<NNL2:TENSOR/%s [%d]:", type_name, rows);
    
	// Handle output truncation for large tensors
    if (rows > max_rows && !full_print) {    
		// Calculate number of elements to skip in the middle
		int skip = rows - 2 * show_rows;
	
        switch(dtype_tensor) {
            case FLOAT64: {
                double* data_t = (double*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                break;
            }
            
            case FLOAT32: {
                float* data_t = (float*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                break;
            }
			
			case INT64: {
                int64_t* data_t = (int64_t*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_INT64_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_INT64_FORMAT, data_t[i]);
                break;
            }
            
            case INT32: {
                int32_t* data_t = (int32_t*)tensor->data;
                for(int i = 0; i < show_rows; i++) printf("\n    " NNL2_INT32_FORMAT, data_t[i]);
                printf("\n    ... (%d elements skipped) ...", skip);
                for(int i = rows - show_rows; i < rows; i++) printf("\n    " NNL2_INT32_FORMAT, data_t[i]);
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_tensor);
                return;
            }
        }
    } else {
		// Print all elements for small tensors or when full_print is requested
        switch(dtype_tensor) {
            case FLOAT64: {
                double* data_t = (double*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_FLOAT64_FORMAT, data_t[i]);
                break;
            }
            
            case FLOAT32: {
                float* data_t = (float*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_FLOAT32_FORMAT, data_t[i]);
                break;
            }
			
			case INT64: {
                int64_t* data_t = (int64_t*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_INT64_FORMAT, data_t[i]);
                break;
            }
            
            case INT32: {
                int32_t* data_t = (int32_t*)tensor->data;
                for(int i = 0; i < rows; i++) 
                    printf("\n    " NNL2_INT32_FORMAT, data_t[i]);
                break;
            }
            
            default: {
                NNL2_TYPE_ERROR(dtype_tensor);
                return;
            }
        }
    }
    
	// Close tensor output format
    printf(">\n");
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif	
}

#endif /** NNL2_1D_H **/
