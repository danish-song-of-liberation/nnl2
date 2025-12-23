#ifndef NNL2_2D_H
#define NNL2_2D_H

/** @brief 
 * Prints the contents of a 2D tensor (matrix) to standard output
 *
 ** @param tensor
 * Pointer to the 2D tensor to be printed 
 *
 ** @param full_print
 * Flag controlling output truncation:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors (show first and last rows/columns)
 *
 ** @note
 * In safety mode, performs extensive validation of input parameters
 *
 * @note
 * Output format includes tensor metadata (type, shape) and formatted data.
 * Truncation can be applied independently to rows and columns.
 *
 ** @example
 * // Print full matrix contents
 * nnl2_print_2d_tensor(my_matrix, true, 10, 10, 3, 5);
 *
 * // Print truncated version for large matrices  
 * nnl2_print_2d_tensor(large_matrix, false, 10, 10, 3, 5);
 *
 **/
void nnl2_print_2d_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t max_cols, int32_t quantity_show_rows, int32_t quantity_show_cols) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

	// Comprehensive input validation in maximal safety mode
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
        
        if (tensor->rank != 2) {
            NNL2_ERROR("Expected 2D tensor, got %dD", tensor->rank);
            return;
        }
    #endif
    
    int rows = tensor->shape[0];
    int cols = tensor->shape[1];
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if (rows <= 0 || cols <= 0) {
            NNL2_ERROR("Invalid tensor shape [%d, %d]", rows, cols);
            return;
        }
    #endif
	
	if (!full_print) {
        if (rows > max_rows && 2 * quantity_show_rows >= rows) {
            NNL2_ERROR("quantity_show_rows (%d) Is too large for tensor rows (%d). Check the correctness of the tensor formatting settings you have specified", quantity_show_rows, rows);
            return;
        }
        
        if (cols > max_cols && 2 * quantity_show_cols >= cols) {
            NNL2_ERROR("quantity_show_cols (%d) Is too large for tensor columns (%d). Check the correctness of the tensor formatting settings you have specified", quantity_show_cols, cols);
            return;
        }
    }
    
    TensorType dtype_tensor = tensor->dtype;
    char* type_name = get_tensortype_name(dtype_tensor);
    
	// Prefix
    printf("#<NNL2:TENSOR/%s [%dx%d]:", type_name, rows, cols);
    
    bool truncate_rows = (rows > max_rows) && !full_print;
    bool truncate_cols = (cols > max_cols) && !full_print;
    
	// Number of rows/columns displayed before and after skipping
    int show_rows = truncate_rows ? quantity_show_rows : rows;
    int show_cols = truncate_cols ? quantity_show_cols : cols;

    switch(dtype_tensor) {
        case FLOAT64: {
            double* data_t = (double*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
					// Concatenate "    " with NNL2_FLOAT64_FORMAT
                    printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                }
				
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
						// Concatenate "    " with NNL2_FLOAT64_FORMAT
                        printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
						// Concatenate "    " with NNL2_FLOAT64_FORMAT
                        printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                    }
					
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
							// Concatenate "    " with NNL2_FLOAT64_FORMAT
                            printf("    " NNL2_FLOAT64_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
			
            break;
        }
        
        case FLOAT32: {
            float* data_t = (float*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
					// Concatenate "    " with NNL2_FLOAT32_FORMAT
                    printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                }
				
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
						// Concatenate "    " with NNL2_FLOAT32_FORMAT
                        printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
						// Concatenate "    " with NNL2_FLOAT32_FORMAT
                        printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                    }
					
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
							// Concatenate "    " with NNL2_FLOAT32_FORMAT
                            printf("    " NNL2_FLOAT32_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
			
            break;
        }
        
		case INT64: {
            int64_t* data_t = (int64_t*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
                    printf("    " NNL2_INT64_FORMAT, data_t[i * cols + j]);
                }
                
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
                        printf("    " NNL2_INT64_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
                        printf("    " NNL2_INT64_FORMAT, data_t[i * cols + j]);
                    }
                    
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
                            printf("    " NNL2_INT64_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
			
            break;
        }
		
        case INT32: {
            int32_t* data_t = (int32_t*)tensor->data;
            
            for (int i = 0; i < show_rows; i++) {
                printf("\n");
                for (int j = 0; j < show_cols; j++) {
                    printf("    " NNL2_INT32_FORMAT, data_t[i * cols + j]);
                }
                
                if (truncate_cols) {
                    printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                    for (int j = cols - show_cols; j < cols; j++) {
                        printf("    " NNL2_INT32_FORMAT, data_t[i * cols + j]);
                    }
                }
            }
            
            if (truncate_rows) {
                printf("\n    ... (%d rows skipped) ...", rows - 2 * show_rows);
                
                for (int i = rows - show_rows; i < rows; i++) {
                    printf("\n");
                    for (int j = 0; j < show_cols; j++) {
                        printf("    " NNL2_INT32_FORMAT, data_t[i * cols + j]);
                    }
                    
                    if (truncate_cols) {
                        printf("    ... (%d cols skipped) ...", cols - 2 * show_cols);
                        for (int j = cols - show_cols; j < cols; j++) {
                            printf("    " NNL2_INT32_FORMAT, data_t[i * cols + j]);
                        }
                    }
                }
            }
            
            break;
        }
        
        default: {
            NNL2_TYPE_ERROR(dtype_tensor);
            return;
        }
    }
    
	// Closing format
    printf(">\n");
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_2D_H **/
