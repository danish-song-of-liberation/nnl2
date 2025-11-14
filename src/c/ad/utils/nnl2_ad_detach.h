#ifndef NNL2_DETACH_H
#define NNL2_DETACH_H

// NNL2

/** @file nnl2_ad_detach.h
 ** @date 2025  
 ** @copyright MIT License
 ** @brief Contains detach function (out-of-place)
 **/
 
/** @brief 
 * Creates a detached copy of a tensor that shares data but disconnects from graph 
 *
 ** @param ad_tensor 
 * Pointer to the original tensor to detach from computational graph
 *
 ** @return nnl2_ad_tensor* 
 * Pointer to the detached tensor, or NULL if operation fails
 *
 ** @note 
 * The detached tensor shares the same data buffer as the original tensor
 *
 ** @note 
 * Modifications to the detached tensor's data will affect the original tensor and vice versa
 *
 ** @see nnl2_ad_tensor_share_data()
 **/
nnl2_ad_tensor* nnl2_ad_detach(nnl2_ad_tensor* ad_tensor) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_detach, ad_tensor is NULL", NULL);
    #endif 
	
	// Create a shared data tensor
	nnl2_ad_tensor* detached = nnl2_ad_tensor_share_data((nnl2_ad_tensor*)ad_tensor);
    if(detached == NULL) {
		NNL2_ERROR("In function nnl2_ad_detach, failed to share original tensor data");
        return NULL;
    }
	
	detached->is_leaf = true;         
    detached->requires_grad = false;
	
	if(detached->name != NULL) {
        free(detached->name);
    }
	
	if(ad_tensor->name != NULL) {
        size_t name_len = strlen(ad_tensor->name) + 16;  // "detached()" + original name + null terminator
        detached->name = malloc(name_len);

        if(detached->name != NULL) {
            snprintf(detached->name, name_len, "detached(%s)", ad_tensor->name);
        }
    } else {
        detached->name = NULL;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
    
    return detached;
}
	
#endif /** NNL2_DETACH_H **/
