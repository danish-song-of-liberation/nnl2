#ifndef NNL2_PRINT_TENSOR_H
#define NNL2_PRINT_TENSOR_H

/** @brief 
 * Universal tensor printing function that routes to appropriate specialized printer
 *
 ** @param tensor
 * Pointer to the tensor to be printed
 *
 ** @param full_print
 * Flag controlling output detail level:
 ** true: Print all elements regardless of tensor size
 ** false: Truncate output for large tensors
 *
 ** @note
 * Automatically selects the appropriate printing function based on tensor rank:
 ** Rank 1: Uses print_1d_tensor() for vector printing
 ** Rank 2: Uses print_2d_tensor() for matrix printing  
 ** Rank 3+: Uses print_huge_tensor() for metadata-only display
 ** Invalid rank: Handles error or returns silently based on safety mode
 *
 ** @note
 * In safety mode, performs validation of input parameters and tensor structure
 *
 ** @example
 * // Print full tensor contents
 * print_tensor(my_tensor, true);
 *
 * // Print truncated version for large tensors
 * print_tensor(large_tensor, false);
 *
 ** @see nnl2_print_1d_tensor
 ** @see nnl2_print_2d_tensor  
 ** @see nnl2_print_huge_tensor
 **/
void nnl2_print_tensor(Tensor* tensor, bool full_print, int32_t max_rows, int32_t max_cols, int32_t show_rows, int32_t show_cols) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	int32_t rank = tensor->rank;
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(rank <= 0) {
			NNL2_ERROR("Invalid rank: %d. Rank should be non-negative", rank);
			return;
		}
	#else 
		 if(rank <= 0) {return;}
	#endif
	else if(rank == 1) {nnl2_print_1d_tensor(tensor, full_print, max_rows, show_rows);}
	else if(rank == 2) {nnl2_print_2d_tensor(tensor, full_print, max_rows, max_cols, show_rows, show_cols);}
	else 			   {nnl2_print_huge_tensor(tensor);}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief
 * Function for quick debugging in C without specifying 
 * all the arguments in nnl2_print_tensor
 *
 ** @param tensor
 * Tensor to print
 */
void nnl2_quick_print_tensor(Tensor* tensor) {
	nnl2_print_tensor(tensor, 
					  false, // Full print?
					  20, // Max rows to print
					  10, // Max cols to print
					  10, // Show rows non in skip 
					  5); // Show cols non in skip
}

#endif /** NNL2_PRINT_TENSOR_H **/
