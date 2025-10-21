#ifndef NNL2_AD_GENS_H
#define NNL2_AD_GENS_H

/** @file nnl2_ad_gens.h
 ** @copyright MIT
 ** @date 2025
 *
 * The file contains functions for creating 
 * AD-tensors, including zeros, ones, full, etc.
 *
 ** Filepath: nnl2/src/c/ad/nnl2_adgens.h
 **/
 
/** @brief 
 * Creates a new automatic differentiation 
 * tensor filled with specified values
 * 
 ** @param shape 
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype 
 * Data type of the tensor elements (nnl2_tensor_type)
 *
 ** @param requires_grad 
 * Boolean flag indicating whether gradient tracking is enabled
 *
 ** @param name 
 * Optional string identifier for the tensor (can be NULL)
 *
 ** @param fill_with 
 * Pointer to value used to initialize tensor elements
 * 
 ** @return nnl2_ad_tensor*
 * Pointer to the newly created nnl2_ad_tensor on success, NULL on failure
 *
 ** @note 
 * If requires_grad is true, a gradient tensor of the same shape is allocated
 *
 ** @note 
 * The created tensor is marked as a leaf node in the computation graph
 * 
 ** @warning 
 * If name is provided, it is copied internally; the original string can be freed
 * 
 ** @see nnl2_free_ad_tensor()
 ** @see nnl2_full()
 ** @see nnl2_empty()
 * *
 ** @example 
 ** @code
 * int32_t shape[] = {2, 3};
 * float fill_value = 1.0f;
 * nnl2_ad_tensor* tensor = nnl2_ad_full(shape, 2, FLOAT32, true, "weight_matrix", &fill_value);
 * if (tensor) {
 *     // Use tensor...
 *     nnl2_free_ad_tensor(tensor);
 * }
 ** @endcode
 **/
nnl2_ad_tensor* nnl2_ad_full(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, void* fill_with) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Allocate memory for the automatic differentiation tensor structure
	nnl2_ad_tensor* ad_tensor = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!ad_tensor) {
			NNL2_MALLOC_ERROR();
			return NULL; 
		}
	#endif
	
	// Create the main data tensor filled with specified values
	ad_tensor->data = nnl2_full(shape, rank, dtype, fill_with);
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if (!ad_tensor->data) {
			NNL2_TENSOR_ERROR("custom");
			free(ad_tensor);
			return NULL; 
		}
	#endif
	
	// If gradient tracking is required, allocate gradient tensor
	if(requires_grad) {
		// Create empty gradient tensor with same shape and type
		ad_tensor->grad = nnl2_empty(shape, rank, dtype);
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			if (!ad_tensor->grad) {
				NNL2_TENSOR_ERROR("non-initialized memory");
				nnl2_free_tensor(ad_tensor->data);
				free(ad_tensor);
				return NULL;
			}
		#endif
	} else {
		ad_tensor->grad = NULL;
	}
	
	// Initialize automatic differentiation properties
	ad_tensor->requires_grad = requires_grad;
	ad_tensor->backward_fn = NULL;
	ad_tensor->roots = NULL;
    ad_tensor->num_roots = 0;
    ad_tensor->grad_computed = NULL;
	ad_tensor->is_leaf = true;
	
	if(name) {
		ad_tensor->name = malloc(strlen(name) + 1);
		
		// Copy name if allocation succeeded, otherwise warn and set to NULL
		if (ad_tensor->name) {
			strcpy(ad_tensor->name, name);
		} else {
			NNL2_WARN("Failed to allocate memory for the AD tensor name (the name will be replaced with NULL)");
			ad_tensor->name = NULL;
		}
	} else {
		// No name provided, set to NULL
		ad_tensor->name = NULL;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
		
	return ad_tensor;
}	

/** @brief 
 * Creates a new automatic differentiation tensor filled with zeros
 * 
 ** @param shape 
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype 
 * Data type of the tensor elements (nnl2_tensor_type)
 *
 ** @param requires_grad 
 * Boolean flag indicating whether gradient tracking is enabled
 *
 ** @param name 
 * Optional string identifier for the tensor (can be NULL)
 * 
 ** @return 
 * Pointer to the newly created nnl2_ad_tensor filled 
 * with zeros on success, NULL on failure
 *
 ** @warning 
 * The caller is responsible for freeing the returned tensor
 * 
 ** @note 
 * The function handles proper type conversion internally
 *
 ** @see nnl2_ad_full()
 ** @see nnl2_ad_ones()
 * 
 ** @example 
 ** @code
 * int32_t shape[] = {3, 4};
 * nnl2_ad_tensor* zeros = nnl2_ad_zeros(shape, 2, FLOAT32, true, "bias_zeros");
 ** @endcode
 **/
nnl2_ad_tensor* nnl2_ad_zeros(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Define zero value as float32 (will be properly converted based on dtype)
	nnl2_float32 zero = 0.0f;
	
	// Create tensor filled with zeros
	nnl2_ad_tensor* ad_tensor = nnl2_ad_full(shape, rank, dtype, requires_grad, name, &zero);
	if(ad_tensor == NULL) {
		NNL2_TENSOR_ERROR("zeros");
		return NULL;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return ad_tensor;
}

/** @brief 
 * Creates a new automatic differentiation tensor filled with ones
 *
 ** @param shape 
 * Array of integers defining the dimensions of the tensor
 *
 ** @param rank 
 * Number of dimensions (length of shape array)
 *
 ** @param dtype 
 * Data type of the tensor elements (nnl2_tensor_type)
 *
 ** @param requires_grad 
 * Boolean flag indicating whether gradient tracking is enabled
 *
 ** @param name 
 * Optional string identifier for the tensor (can be NULL)
 * 
 ** @return 
 * Pointer to the newly created nnl2_ad_tensor 
 * filled with ones on success, NULL on failure
 * 
 ** @warning 
 * The caller is responsible for freeing the returned tensor
 * 
 ** @note 
 * The function handles proper type conversion internally
 *
 ** @see nnl2_ad_full()
 ** @see nnl2_ad_zeros()
 * 
 ** @example 
 ** @code
 * int32_t shape[] = {2, 2};
 * nnl2_ad_tensor* ones = nnl2_ad_ones(shape, 2, FLOAT32, false, "identity_matrix");
 ** @endcode
 **/
nnl2_ad_tensor* nnl2_ad_ones(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
    // Define one value as float32 (will be properly converted based on dtype)
	nnl2_float32 one = 1.0f;
	
	// Create tensor filled with ones 
	nnl2_ad_tensor* ad_tensor = nnl2_ad_full(shape, rank, dtype, requires_grad, name, &one);
	if(ad_tensor == NULL) {
		NNL2_TENSOR_ERROR("ones");
		return NULL;
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return ad_tensor;
}
 
#endif /** NNL2_AD_GENS_H **/
