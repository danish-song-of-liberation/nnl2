#ifndef NNL2_TRANS_INPLACE_H
#define NNL2_TRANS_INPLACE_H

/** @brief
 * Transposes a matrix in place (naive implementation)
 *
 ** @warning
 * Not thread-safe
 *
 ** @param tensor
 * Pointer to a tensor for transposition
 *
 ** @see product
 ** @see https://en.wikipedia.org/wiki/Dont_repeat_yourself 
 **/
void naive_transposeinplace(Tensor* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		int32_t tensor_rank = tensor->rank;
		if(tensor_rank < 2 || tensor_rank > 2) {
			NNL2_FATAL("Tensor have an incorrect rank: %d (expected 2)", tensor_rank);
			return;
		}
	#endif
	
	// Calculating the total number of elements in a tensor
	size_t total_elems = product(tensor->shape, tensor->rank);
	
	int* shape = tensor->shape;
	
	int rows = shape[0];
	int cols = shape[1];
	
	switch(tensor->dtype) {
		case FLOAT64: {
			size_t total_bytes = total_elems * sizeof(double);
			
			// Allocating memory for temporary storage of the transposed matrix
			double* trans_data = (double*)malloc(total_bytes);
			double* cast_data = (double*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) {
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case FLOAT32: {
			size_t total_bytes = total_elems * sizeof(float);
			
			// Allocating memory for temporary storage of the transposed matrix
			float* trans_data = (float*)malloc(total_bytes);
			float* cast_data = (float*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		case INT32: {
			size_t total_bytes = total_elems * sizeof(int32_t);
			
			// Allocating memory for temporary storage of the transposed matrix
			int32_t* trans_data = (int32_t*)malloc(total_bytes);
			int32_t* cast_data = (int32_t*)tensor->data;
			
			if (trans_data == NULL) {
				NNL2_ERROR("Memory allocation failed");
				return;
			}
			
			// Matrix transposition: element [i][j] -> [j][i]
			for(int i = 0; i < rows; i++) {
				for(int j = 0; j < cols; j++) { 
					int orig_index = i * cols + j;   // Index in the original matrix
					int trans_index = j * rows + i;  // Index in the transposed matrix
					
					trans_data[trans_index] = cast_data[orig_index];
				}
			}
			
			// Copying the result back to the original tensor
			memcpy(cast_data, trans_data, total_bytes);
			free(trans_data);
			
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype);
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for transposeinplace operation
 * @details
 * Array follows the common backend registration pattern for in-place transpose
 * operations. Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for in-place transpose
 * 
 * @see nnl2_naive
 * @see naive_transposeinplace
 */
Implementation transposeinplace_backends[] = {
	REGISTER_BACKEND(naive_transposeinplace, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/**
 * @brief Function pointer for transposeinplace operation
 * @ingroup backend_system 
 */
transposeinplacefn transposeinplace;

/** 
 * @brief Makes the transposeinplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(transposeinplace);

/** 
 * @brief Sets the backend for transposeinplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for transposeinplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_transposeinplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(transposeinplace_backends, transposeinplace, backend_name, current_backend(transposeinplace));
}

/** 
 * @brief Gets the name of the active backend for transposeinplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_transposeinplace_backend() {
	return current_backend(transposeinplace);
}

/** 
 * @brief Function declaration for getting all available transposeinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(transposeinplace);

/**
 * @brief Function declaration for getting the number of available transposeinplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(transposeinplace);

#endif /** NNL2_TRANS_INPLACE_H **/
