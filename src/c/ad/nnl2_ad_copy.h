#ifndef NNL2_AD_COPY_H
#define NNL2_AD_COPY_H

/** @file nnl2_ad_copy.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains function with copy of AD tensors 
 **/

/** @brief 
 * Creates a deep copy of an autodiff tensor with optional type conversion
 * 
 ** @param ad_tensor 
 * Pointer to the source autodiff tensor to copy
 *
 ** @param dtype 
 * Target data type for the copied tensor data (can be same as source)
 *
 ** @return 
 * Pointer to the newly allocated autodiff tensor copy, or NULL if failure
 */
static inline nnl2_ad_tensor* nnl2_ad_copy(nnl2_ad_tensor* restrict ad_tensor, nnl2_tensor_type dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_copy, ad_tensor is NULL", NULL);
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor->data, "In function nnl2_ad_copy, ad_tensor data is NULL", NULL);
	#endif
	
	nnl2_ad_tensor* tensor_copy = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
	if(!tensor_copy) {
        NNL2_MALLOC_ERROR();
        return NULL;
    }
	
	// Deep copy
	tensor_copy->data = nnl2_copy(ad_tensor->data, dtype);
	if(!tensor_copy->data) {
        NNL2_ERROR("In function nnl2_ad_copy, failed to copy data tensor");
        free(tensor_copy);
        return NULL;
    }
	
	// Copy gradient if initialized
	if(ad_tensor->grad_initialized && ad_tensor->grad) {
		tensor_copy->grad = nnl2_copy(ad_tensor->grad, dtype);
		tensor_copy->grad_initialized = true;
	} else {
		tensor_copy->grad = NULL;
		tensor_copy->grad_initialized = false;
	}

	// Copy basic metadata
    tensor_copy->ts_type = nnl2_type_ad;
    tensor_copy->is_leaf = ad_tensor->is_leaf;
    tensor_copy->requires_grad = ad_tensor->requires_grad;
    tensor_copy->backward_fn = ad_tensor->backward_fn; 
    tensor_copy->magic_number = TENSOR_MAGIC_ALIVE;
	
	// Shallow copy of graph structure
    if(ad_tensor->num_roots > 0 && ad_tensor->roots) {
        tensor_copy->num_roots = ad_tensor->num_roots;
        tensor_copy->roots = (nnl2_ad_tensor**)malloc(ad_tensor->num_roots * sizeof(*tensor_copy->roots));
        if(!tensor_copy->roots) {
            NNL2_MALLOC_ERROR();
            nnl2_free_tensor(tensor_copy->data); 
			if(tensor_copy->grad) nnl2_free_tensor(tensor_copy->grad); 
			free(tensor_copy);
            return NULL;
        }

        for(size_t i = 0; i < tensor_copy->num_roots; ++i) tensor_copy->roots[i] = ad_tensor->roots[i];
    } else {
        tensor_copy->num_roots = 0;
        tensor_copy->roots = NULL;
    }
	
	// Copy name if present
	if(ad_tensor->name) {
        size_t len = strlen(ad_tensor->name) + 1;
        tensor_copy->name = malloc(len);
        if (tensor_copy->name) {
            memcpy(tensor_copy->name, ad_tensor->name, len);
        } else {
            NNL2_MALLOC_ERROR();
		}
    } else {
        tensor_copy->name = NULL;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return tensor_copy;
}

#endif
