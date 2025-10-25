#ifndef NNL2_EMPTY_H
#define NNL2_EMPTY_H

/** @brief
 * Creates a new tensor without initializing the data
 *
 * This function allocates memory for the Tensor structure and its data,
 * using the provided shape and data type. The data memory is not (!) initialized
 *
 * And also, the tensor has strides fields that will be initialized inside
 *
 ** @param shape
 * A pointer to an array of integers representing the tensor's shape
 *
 ** @param rank
 * The number of dimensions of the tensor
 *
 ** @param dtype
 * Tensor data type (TensorType)
 *
 ** @return 
 * pointer to a new tensor or NULL in case of an error
 *
 ** @details
 * The function firstly:
 *
 *
 *** Checks the input parameters for correctness (namely shape, rank, dtype)
 ** Then
 *** Allocates memory for tensor structure
 ** Then
 *** Allocates memory for the shape array and copies the data into it
 ** Then
 *** Calculates the total number of elements and strides
 ** Then
 *** Checks for multiplication overflow to prevent size_t 
 *** overflow when calculating total memory size
 ** Then 
 *** Allocates aligned memory for the tensor data (tensor->data)
 ** Finally
 *** Returns a pointer to the created tensor
 *
 *
 ** @code
 * int shape[] = {2, 3, 4};
 * Tensor* my_tensor = nnl2_empty(shape, 3, FLOAT32);
 * if (my_tensor == NULL) {
 *     // Handle error
 * }
 ** @endcode
 ** @note
 * The function performs comprehensive error checking including:
 ** - NULL pointer validation
 ** - Parameter range validation  
 ** - Memory allocation failure handling
 ** - Multiplication overflow protection
 ** - Type safety checks
 *
 ** @note
 * In debug mode, it can output additional information to the console
 *
 ** @note
 * Сan perform additional checks at a high level of safety
 *
 ** @warning
 * Do not forget to free the memory allocated for the tensor using free_tensor after using it
 *
 ** @see free_tensor
 ** @see get_dtype_size
 ** @see product
 ** @see ALLOC_ALIGNED
 ** @see NNL2_DEBUG_MODE
 ** @see NNL2_SAFETY_MODE
 ** @see NNL2_STRIDES_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_SECOND_LAST_DIM_INDEX
 ** @see NNL2_STRIDES_NEXT_DIM_INDEX
 *
 ** @exception[out_of_memory] 
 * If memory allocation fails for tensor structure, shape array, or data
 *
 ** @exception[invalid_argument]
 * If shape is NULL, rank <= 0, invalid dtype, or any dimension <= 0
 *
 ** @exception[overflow_error]
 * If the calculated total size would overflow size_t
 **/
Tensor* nnl2_naive_empty(const int32_t* shape, const int32_t rank, const TensorType dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		NNL2_FUNC_ENTER();
	#endif

	// Checks the input parameters for correctness
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		if (shape == NULL) {
			NNL2_ERROR("Bad shape pointer (NULL)"); 
			return NULL;
		}	

		if (rank <= 0) {
			NNL2_ERROR("Bad rank (%d). Rank must be positive", rank);
			return NULL;
		}
		
		if (dtype < 0 || dtype >= NUM_TENSOR_TYPES) {
			NNL2_ERROR("Bad tensor type (%d)", dtype);
			return NULL;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		for (int i = 0; i < rank; i++) {
			if (shape[i] <= 0) {
				NNL2_ERROR("Bad shape dimension at %d (%d). Dimensions must be positive", i, shape[i]);
				return NULL;
			}
		}
	#endif
	
	// Allocating memory for tensor structure

	Tensor* tensor = malloc(sizeof(Tensor));
	
	if (tensor == NULL) {
        NNL2_ERROR("Failed to allocate memory for Tensor structure");
        return NULL;
    }
	
	tensor->rank = rank;
	tensor->dtype = dtype;
	tensor->is_view = false;
	tensor->magic_number = TENSOR_MAGIC_ALIVE;
	tensor->ts_type = nnl2_type_ts;
	
	// Allocates memory for the shape array and copies the data into it
	
	size_t total_elems = 1;
	tensor->shape = malloc(rank * sizeof(int));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->shape == NULL) {
			NNL2_ERROR("Failed to allocate memory for shape");
			free(tensor);
			return NULL;
		}
	#endif
	
	memcpy(tensor->shape, shape, rank * sizeof(int));
	
	tensor->strides = malloc(rank);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (tensor->strides == NULL) {
			NNL2_ERROR("Failed to allocate memory for strides");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	// Initializes the step for the last dimension
	// Stride for the innermost (last) dimension is 1
	tensor->strides[rank - NNL2_STRIDES_LAST_DIM_INDEX] = 1;
	total_elems *= shape[rank - NNL2_STRIDES_LAST_DIM_INDEX];
	
	// Going from the second-to-last dimension to the first (from left to right in shape)
	for(int i = rank - NNL2_STRIDES_SECOND_LAST_DIM_INDEX; i >= 0; i--) {
		tensor->strides[i] = tensor->strides[i + NNL2_STRIDES_NEXT_DIM_INDEX] * shape[i + NNL2_STRIDES_NEXT_DIM_INDEX];
		total_elems *= shape[i];
	}
	
	size_t type_size = get_dtype_size(dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		// Check for multiplication overflow
		if (type_size != 0 && total_elems > SIZE_MAX / type_size) {
			NNL2_ERROR("Tensor size too large, multiplication overflow (elems: %zu, type size: %zu)", total_elems, type_size);
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif

	size_t total_size = total_elems * type_size;
	
	// Allocates aligned memory for tensor data (tensor->data)
	
	void* data;
	
	ALLOC_ALIGNED(data, (size_t)TENSOR_MEM_ALIGNMENT, total_size);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (data == NULL) {
			NNL2_ERROR("Failed to allocate aligned memory");
			free(tensor->shape);
			free(tensor);
			return NULL;
		}
	#endif
	
	tensor->data = data;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
			NNL2_DEBUG("Created empty tensor: %dD, elems=%zu, size=%zu bytes", rank, total_elems, total_size);
		#endif
	
		NNL2_FUNC_EXIT();
	#endif
	
	return tensor;
}

/** @ingroup backend_system
 ** @brief Backend implementations for empty tensor creation
 ** @details
 * Array follows the common backend registration pattern
 * Currently register backends:
 *  - nnl2_naive_empty: Basic reference implementation
 *
 ** @see REGISTER_BACKEND
 ** @see NAIVE_BACKEND_NAME
 ** @see nnl2_naive_empty
 ** @see nnl2_naive
 **/
Implementation nnl2_empty_backends[] = {
	REGISTER_BACKEND(nnl2_naive_empty, nnl2_naive, NAIVE_BACKEND_NAME)
};

/**
 * @brief Function pointer for the active empty() backend
 * @ingroup backend_system 
 */
fn_empty nnl2_empty;

/** 
 * @brief Sets the backend for empty tensor creation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate
 * @see SET_BACKEND_BY_NAME
 * @see ESET_BACKEND_BY_NAME
 */
void nnl2_set_empty_backend(const char* backend_name) {
    SET_BACKEND_BY_NAME(nnl2_empty_backends, nnl2_empty, backend_name);
}

/** 
 * @brief Creates an empty static string for manual backend work
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(nnl2_empty);

/** 
 * @brief Gets the name of the active backend for empty()
 * @ingroup backend_system
 * @return Name of the current backend
 * @see CURRENT_BACKEND
 */
const char* nnl2_get_empty_backend() {
	return CURRENT_BACKEND(nnl2_empty);
}

/** 
 * @brief Function declaration for getting all `empty` available backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(nnl2_empty);

/**
 * @brief Function declaration for getting the number of all `empty` backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(nnl2_empty);

#endif /** NNL2_EMPTY_H **/
