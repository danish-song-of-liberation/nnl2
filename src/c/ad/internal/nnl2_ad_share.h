#ifndef NNL2_AD_SHARE_H
#define NNL2_AD_SHARE_H

// NNL2

/** @file nnl2_ad_share.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains function for take share memory of AD-tensor
 **/

/** @brief 
 * Creates a new tensor that shares data with the original tensor (view operation)
 *
 ** @param original 
 * Pointer to the original tensor to share data from
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the new shared tensor, or NULL if allocation fails
 */
nnl2_ad_tensor* nnl2_ad_tensor_share_data(nnl2_ad_tensor* original) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(original, "In function nnl2_ad_tensor_share_data, original is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(original->data, "In function nnl2_ad_tensor_share_data, original->data is NULL", NULL);
    #endif 
	
	nnl2_ad_tensor* shared = (nnl2_ad_tensor*)malloc(sizeof(nnl2_ad_tensor));
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(shared == NULL) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif
	
	// Copy the entire tensor structure from original to shared
	if(!memcpy(shared, original, sizeof(nnl2_ad_tensor))) {
		NNL2_ERROR("Failed to copy memory in function nnl2_ad_tensor_share_data from shared to original");
		return NULL;
	}
	
	// Reset autodiff-specific fields
	shared->is_leaf = false;
    shared->grad = NULL;
    shared->grad_initialized = false;  
    shared->backward_fn = NULL;
    shared->roots = NULL;
    shared->num_roots = 0;
    shared->visited_gen = 0;
    shared->extra_multiplier = 1.0f;
    shared->extra_bool = false;
    shared->extra_correspondence = NULL;
	
	// Create a descriptive name for the shared tensor view
	if(original->name != NULL) {
        size_t name_len = strlen(original->name) + 16;  // "view()" + original name + null terminator
        shared->name = (char*)malloc(name_len);
        if(shared->name != NULL) {
            snprintf(shared->name, name_len, "view(%s)", original->name);
        }
    } else {
        shared->name = NULL;  // No name if original had no name
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
	
	return shared;
}

#endif /** NNL2_AD_SHARE_H **/
