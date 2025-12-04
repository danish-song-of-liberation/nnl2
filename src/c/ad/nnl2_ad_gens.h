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
 
void nnl2_free_ad_tensor(nnl2_ad_tensor* ad_tensor);
 
/** @brief 
 * Creates a new automatic differentiation tensor without initializing data
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
 ** @return nnl2_ad_tensor*
 * Pointer to the newly created nnl2_ad_tensor on success, NULL on failure
 *
 ** @note 
 * If requires_grad is true, a gradient tensor of the same shape is allocated
 *
 ** @note 
 * The created tensor is marked as a leaf node in the computation graph
 * The tensor data is uninitialized and must be filled separately
 * 
 ** @warning 
 * If name is provided, it is copied internally; the original string can be freed
 * 
 ** @see nnl2_free_ad_tensor()
 ** @see nnl2_ad_full()
 ** @see nnl2_ad_zeros()
 ** @see nnl2_ad_ones()
 *
 ** @example 
 ** @code
 * int32_t shape[] = {2, 3};
 * nnl2_ad_tensor* tensor = nnl2_ad_empty(shape, 2, FLOAT32, true, "uninitialized_matrix");
 * if (tensor) {
 *     // Manually initialize data...
 *     nnl2_free_ad_tensor(tensor);
 * }
 ** @endcode
 **/
nnl2_ad_tensor* nnl2_ad_empty(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
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
    
    // Create empty data tensor (uninitialized)
    ad_tensor->data = nnl2_empty(shape, rank, dtype);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!ad_tensor->data) {
            NNL2_TENSOR_ERROR("empty");
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
                NNL2_TENSOR_ERROR("zeros");
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
    ad_tensor->is_leaf = true;
	ad_tensor->magic_number = TENSOR_MAGIC_ALIVE;
	ad_tensor->grad_initialized = false;
	ad_tensor->ts_type = nnl2_type_ad;
	ad_tensor->extra_correspondence = NULL;
	ad_tensor->extra_field = NULL;
	ad_tensor->extra_free = NULL;
    
    if(name && name[0] != '\0') {
        ad_tensor->name = malloc(strlen(name) + 1);
        
        // Copy name if allocation succeeded, otherwise warn and set to NULL
        if (ad_tensor->name) {
            strcpy(ad_tensor->name, name);
        } else {
            NNL2_WARN("Failed to allocate memory for the AD tensor name (the name will be replaced with NULL)");
            ad_tensor->name = NULL;
        }
    } else {
        // No name provided, set to unnamed
        ad_tensor->name = malloc(strlen("unnamed") + 1);
		
		if (ad_tensor->name) {
			strcpy(ad_tensor->name, "unnamed");
		} else {
			NNL2_WARN("Failed to allocate memory for the AD tensor name (the name will be replaced with NULL)");
			ad_tensor->name = NULL;
		}
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
        
    return ad_tensor;
}

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
 ** @see nnl2_ad_empty()
 ** @see nnl2_ad_zeros()
 ** @see nnl2_ad_ones()
 *
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
    
    // Create empty AD tensor using the new function
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!ad_tensor) {
            NNL2_TENSOR_ERROR("empty");
            return NULL; 
        }
    #endif
    
    // Fill the tensor data with specified value
    if (!inplace_fill(ad_tensor->data, fill_with, dtype)) {
        NNL2_TENSOR_ERROR("custom");
        nnl2_free_ad_tensor(ad_tensor);
        return NULL;
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
 ** @see nnl2_ad_empty()
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
    
    void* zero_value = nnl2_get_zero_value(dtype);
    if (!zero_value) {
        NNL2_TYPE_ERROR(dtype);
        return NULL;
    }
    
    // Create tensor filled with zeros
    nnl2_ad_tensor* ad_tensor = nnl2_ad_full(shape, rank, dtype, requires_grad, name, zero_value);
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
 ** @see nnl2_ad_empty()
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
    
    void* one_value = nnl2_get_one_value(dtype);
    if (!one_value) {
        NNL2_TYPE_ERROR(dtype);
        return NULL;
    }
    
    // Create tensor filled with ones 
    nnl2_ad_tensor* ad_tensor = nnl2_ad_full(shape, rank, dtype, requires_grad, name, one_value);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("ones");
        return NULL;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}

/** @brief 
 * Creates a new AD tensor and fills it with random values from the specified range
 *
 ** @param shape 
 * Pointer to an array specifying tensor dimensions
 *
 ** @param rank 
 * Number of dimensions in the tensor
 *
 ** @param dtype 
 * Data type of the tensor elements
 *
 ** @param requires_grad 
 * Whether to track this tensor in the AD graph
 *
 ** @param name 
 * Optional tensor name for debugging or inspection
 *
 ** @param from 
 * Pointer to the lower bound of the random range
 *
 ** @param to 
 * Pointer to the upper bound of the random range
 *
 ** @return 
 * Pointer to a newly created AD tensor filled with random values
 * Returns NULL if allocation fails or input validation fails
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If from is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If to is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mode_moderate+]
 * If shape is NULL
 *
 ** @exception NNL2Error
 * If created empty tensor in ```nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(...)``` is NULL (failed)
 *
 ** @see randn_inplace 
 ** @see nnl2_ad_empty
 ** @see nnl2_ad_tensor
 **/
nnl2_ad_tensor* nnl2_ad_uniform(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, void* from, void* to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(from, "In function nnl2_ad_randn, void* from is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(to, "In function nnl2_ad_randn, void* to is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape, "In function nnl2_ad_randn, int32_t* shape is NULL", NULL);
	#endif
    
    // Create empty tensor
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("empty");
        return NULL;
    }
	
	uniform_inplace(ad_tensor->data, from, to);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}

nnl2_ad_tensor* nnl2_ad_rand(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape, "In function nnl2_ad_randn, int32_t* shape is NULL", NULL);
	#endif
    
    // Create empty tensor
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("empty");
        return NULL;
    }
	
	rand_inplace(ad_tensor -> data);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}

nnl2_ad_tensor* nnl2_ad_randn(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, double mean, double std) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape, "In function nnl2_ad_randn, int32_t* shape is NULL", NULL);
	#endif
    
    // Create empty tensor
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("empty");
        return NULL;
    }
	
	randn_inplace(ad_tensor -> data, mean, std);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}

/** @brief 
 * Creates a new AD tensor and initializes it using Xavier initialization
 *
 ** @param shape 
 * Pointer to an array specifying tensor dimensions
 *
 ** @param rank 
 * Number of dimensions in the tensor
 *
 ** @param dtype 
 * Data type of the tensor elements
 *
 ** @param requires_grad 
 * Whether to track this tensor in the AD graph
 *
 ** @param name 
 * Optional tensor name for debugging or inspection
 *
 ** @param in 
 * Number of input units
 *
 ** @param out 
 * Number of output units
 *
 ** @param gain 
 * Scaling factor for the initialization
 *
 ** @param distribution 
 * Distribution type for initialization (6 for uniform, 2 for normal)
 *
 ** @return 
 * Pointer to a newly created AD tensor initialized with Xavier method
 * Returns NULL if allocation fails or input validation fails
 *
 ** @exception NNL2Error [nnl2_safety_mode_moderate+]
 * If shape is NULL
 *
 ** @exception NNL2Error
 * If created empty tensor in ```nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(...)``` is NULL (failed)
 *
 ** @see xavier_inplace 
 ** @see nnl2_ad_empty
 ** @see nnl2_ad_tensor
 **/
nnl2_ad_tensor* nnl2_ad_xavier(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, int in, int out, float gain, float distribution) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape, "In function nnl2_ad_randn, int32_t* shape is NULL", NULL);
	#endif
    
    // Create empty tensor
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("empty");
        return NULL;
    }
	
	xavier_inplace(ad_tensor->data, in, out, gain, distribution);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}

/** @brief 
 * Creates a new AD tensor and initializes it using Kaiming (He) initialization
 *
 ** @param shape 
 * Pointer to an array specifying tensor dimensions
 *
 ** @param rank 
 * Number of dimensions in the tensor
 *
 ** @param dtype 
 * Data type of the tensor elements
 *
 ** @param requires_grad 
 * Whether to track this tensor in the AD graph
 *
 ** @param name 
 * Optional tensor name for debugging or inspection
 *
 ** @param fan_in 
 * Number of input units
 *
 ** @param fan_out 
 * Number of output units
 *
 ** @param gain 
 * Scaling factor for the initialization (usually sqrt(2.0) for ReLU)
 *
 ** @param distribution 
 * Distribution type for initialization (6 for uniform, 2 for normal)
 *
 ** @param mode 
 * Mode of initialization: 
 *  0 = "fan_in" (default), 
 *  1 = "fan_out",
 *  2 = "fan_avg" (average of fan_in and fan_out)
 *
 ** @return 
 * Pointer to a newly created AD tensor initialized with Kaiming method
 * Returns NULL if allocation fails or input validation fails
 *
 ** @exception NNL2Error [nnl2_safety_mode_moderate+]
 * If shape is NULL
 *
 ** @exception NNL2Error
 * If created empty tensor in ```nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(...)``` is NULL (failed)
 *
 ** @see nnl2_naive_kaiming_inplace 
 ** @see nnl2_ad_empty
 ** @see nnl2_ad_tensor
 **/
nnl2_ad_tensor* nnl2_ad_kaiming(int32_t* shape, int rank, nnl2_tensor_type dtype, bool requires_grad, char* name, int fan_in, int fan_out, float gain, float distribution, int mode) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape, "In function nnl2_ad_kaiming, int32_t* shape is NULL", NULL);
    #endif
    
    // Check for integer data type
    if(dtype == INT32) {
        NNL2_FATAL("INT32 can't be used for Kaiming initialization");
        return NULL;
    }
    
    // Create empty tensor
    nnl2_ad_tensor* ad_tensor = nnl2_ad_empty(shape, rank, dtype, requires_grad, name);
    if(ad_tensor == NULL) {
        NNL2_TENSOR_ERROR("empty");
        return NULL;
    }
    
    // Apply Kaiming initialization
    nnl2_naive_kaiming_inplace(ad_tensor->data, fan_in, fan_out, gain, distribution, mode);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return ad_tensor;
}
 
/** @brief 
 * Frees all memory associated with an automatic differentiation tensor
 * 
 ** @param ad_tensor 
 * Pointer to the nnl2_ad_tensor to be freed
 * 
 ** @see nnl2_free_tensor()
 ** @see nnl2_ad_full()
 ** @see nnl2_ad_zeros()
 ** @see nnl2_ad_ones()
 ** @see nnl2_ad_empty()
 **
 ** @example 
 ** @code
 * nnl2_ad_tensor* tensor = nnl2_ad_zeros(shape, 2, FLOAT32, true, "example");
 * // Use tensor...
 * nnl2_free_ad_tensor(tensor); // Clean up memory
 * tensor = NULL; // Good practice to avoid dangling pointers
 ** @endcode
 **/
void nnl2_free_ad_tensor(nnl2_ad_tensor* ad_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	if(!ad_tensor->magic_number) {
        return;
	}
	
	if (ad_tensor->magic_number != TENSOR_MAGIC_ALIVE) return;
	ad_tensor->magic_number = TENSOR_MAGIC_FREED;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("Freeing %p tensor (name: %s)", ad_tensor, ad_tensor -> name);
    #endif
    
    // Free the main data tensor
    if (ad_tensor->data) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_DEBUG("Freeing AD-tensor data at %x", ad_tensor->data);
        #endif
    
        nnl2_free_tensor(ad_tensor->data);
    }
    
    // Free the gradient tensor
    if (ad_tensor->grad) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_DEBUG("Freeing AD-tensor gradient at %x", ad_tensor->grad);
        #endif
        
        nnl2_free_tensor(ad_tensor->grad);
    }
    
    // Free the name string
    if (ad_tensor->name) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_DEBUG("Freeing AD-tensor name at %x", ad_tensor->name);
        #endif
        
        free(ad_tensor->name);
    }
    
    // Free the roots array
    if (ad_tensor->roots) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_DEBUG("Freeing AD-tensor roots at %x", ad_tensor->roots);
        #endif
        
        free(ad_tensor->roots);
    }
	
	// Free extra_field if there's a cleanup function
    if (ad_tensor->extra_free && ad_tensor->extra_field) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_DEBUG("Freeing AD-tensor extra_field at %x with custom free function", ad_tensor->extra_field);
        #endif
        
        ad_tensor->extra_free(ad_tensor->extra_field);
		
		ad_tensor->extra_field = NULL;
		ad_tensor->extra_free = NULL;
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_DEBUG("Freeing AD-tensor at %x", ad_tensor);
    #endif
        
    // Free the main AD tensor structure
    free(ad_tensor);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}
 
#endif /** NNL2_AD_GENS_H **/
