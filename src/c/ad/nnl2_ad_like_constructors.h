#ifndef NNL2_AD_LIKE_CONSTRUCTORS
#define NNL2_AD_LIKE_CONSTRUCTORS

/** @file nnl2_ad_like_costructors.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains functions like zeros_like, ones_like, etc.
 **/

/** @brief 
 * Creates a new empty tensor with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created empty tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_empty(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @see nnl2_ad_empty
 **/
nnl2_ad_tensor* nnl2_ad_empty_like(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_empty_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_empty_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
	// Tensor allocation
	nnl2_ad_tensor* result = nnl2_ad_empty(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("empty");
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief 
 * Creates a new tensor filled with zeros with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created zeros tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_zeros(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @see nnl2_ad_zeros
 **/
nnl2_ad_tensor* nnl2_ad_zeros_like(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_zeros_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_zeros_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
	// Tensor allocation with zeros
	nnl2_ad_tensor* result = nnl2_ad_zeros(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("zeros");
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief 
 * Creates a new tensor filled with ones with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created ones tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_ones(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @see nnl2_ad_ones
 **/
nnl2_ad_tensor* nnl2_ad_ones_like(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_ones_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_ones_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
	// Tensor allocation with ones
	nnl2_ad_tensor* result = nnl2_ad_ones(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("ones");
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result; 
}

/** @brief 
 * Creates a new tensor filled with specified value with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @param filler 
 * Pointer to the value used to fill the tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created filled tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If filler is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_full(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @see nnl2_ad_full
 **/
nnl2_ad_tensor* nnl2_ad_full_like(nnl2_ad_tensor* ad_tensor, void* filler) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_full_like, ad_tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(filler, "In function nnl2_ad_full_like, void* filler is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_full_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
	// Tensor allocation with filler value
	nnl2_ad_tensor* result = nnl2_ad_full(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, filler);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("full");
			return NULL;
		}
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

/** @brief 
 * Creates a new tensor with random values in range [0, 1] with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created random tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_randn(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If unsupported data type is encountered
 *
 ** @see nnl2_ad_randn
 **/
nnl2_ad_tensor* nnl2_ad_rand_like(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_rand_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_rand_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
    switch(ad_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64 from = 0.0;
            nnl2_float64 to = 1.0;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
	
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
	
            return result;
        }
		
        case FLOAT32: {
            nnl2_float32 from = 0.0f;
            nnl2_float32 to = 1.0f;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
            return result;
        }
		
        case INT32: {
            nnl2_int32 from = 0;
            nnl2_int32 to = 1;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
            return result;
        }
		
        default: {
            NNL2_TYPE_ERROR(ad_tensor->data->dtype);
            return NULL;
        }
    }
}

/** @brief 
 * Creates a new tensor with random values in range [-1, 1] with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created random tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_randn(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If unsupported data type is encountered
 *
 ** @see nnl2_ad_randn
 **/
nnl2_ad_tensor* nnl2_ad_randn_like(nnl2_ad_tensor* ad_tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_randn_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_randn_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
    switch(ad_tensor->data->dtype) {
        case FLOAT64: {
            nnl2_float64 from = -1.0;
            nnl2_float64 to = 1.0;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
            return result;
        }
		
        case FLOAT32: {
            nnl2_float32 from = -1.0f;
            nnl2_float32 to = 1.0f;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
            return result;
        }
		
        case INT32: {
            nnl2_int32 from = -1;
            nnl2_int32 to = 1;
			
			nnl2_ad_tensor* result = nnl2_ad_randn(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, &from, &to);
			
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
				if(!result) {
					NNL2_TENSOR_ERROR("randn");
					return NULL;
				}
			#endif
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
				NNL2_FUNC_EXIT();
			#endif
			
            return result;
        }
		
        default: {
            NNL2_TYPE_ERROR(ad_tensor->data->dtype);
            return NULL;
        }
    }
}

/** @brief 
 * Creates a new tensor with Xavier/Glorot initialization with the same characteristics as the input tensor
 *
 ** @param ad_tensor 
 * Pointer to the reference tensor used as template for the new tensor
 *
 ** @param in
 * Number of input neurons for Xavier initialization
 *
 ** @param out
 * Number of output neurons for Xavier initialization
 *
 ** @param gain
 * Scaling factor for the initialization
 *
 ** @param dist
 * Distribution parameter for initialization
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the newly created Xavier-initialized tensor, or NULL if error occurred
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_min+]
 * If result (see ```nnl2_ad_tensor* result = nnl2_ad_xavier(...)```) is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mod_moderate+]
 * If ad_tensor->shape is NULL
 *
 ** @see nnl2_ad_xavier
 **/
nnl2_ad_tensor* nnl2_ad_xavier_like(nnl2_ad_tensor* ad_tensor, int in, int out, float gain, float dist) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_xavier_like, ad_tensor is NULL", NULL);
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data->shape, "In function nnl2_ad_xavier_like, ad_tensor->data->shape is NULL", NULL);
	#endif
	
	// Tensor allocation with Xavier initialization
	nnl2_ad_tensor* result = nnl2_ad_xavier(ad_tensor->data->shape, ad_tensor->data->rank, ad_tensor->data->dtype, ad_tensor->requires_grad, ad_tensor->name, in, out, gain, dist);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!result) {
			NNL2_TENSOR_ERROR("xavier");
			return NULL;
		}
	#endif
			
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#endif /** NNL2_AD_LIKE_CONSTRUCTORS **/
