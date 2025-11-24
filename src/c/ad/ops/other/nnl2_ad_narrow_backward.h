#ifndef NNL2_AD_NARROW_BACKWARD_H
#define NNL2_AD_NARROW_BACKWARD_H

/** @file nnl2_ad_narrow_backward.h
 ** @date 2025
 ** @copyright MIT
 ** @brief AD Backward for narrow operation
 **/

/** @brief 
 * Computes the gradient of the narrow operation in reverse mode AD
 *
 ** @param new_tensor 
 * The output tensor from the narrow operation (narrowed view)
 *
 ** @param origin_tensor 
 * The input tensor to the narrow operation (original tensor)
 *
 ** @param dim 
 * The dimension along which to narrow
 *
 ** @param start 
 * The starting index of the narrow operation
 *
 ** @param len 
 * The length of the narrow operation
 *
 ** @warning
 * Do not call the function directly. This is an internal function for AD
 *
 ** @note
 * Since narrow creates a view (O(1) operation), gradient is propagated
 * only to the corresponding region in the original tensor's gradient
 *
 ** @see nnl2_narrow()
 ** @see nnl2_add_inplace()
 ** @see nnl2_free_tensor()
 **/
static NNL2_FORCE_INLINE void nnl2_ad_reverse_derivative_narrow(nnl2_ad_tensor* new_tensor, nnl2_ad_tensor* origin_tensor, uint8_t dim, int32_t start, int32_t len) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(new_tensor, "In function nnl2_ad_reverse_derivative_narrow, new_tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(origin_tensor, "In function nnl2_ad_reverse_derivative_narrow, origin_tensor is NULL");
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        NNL2_CHECK_NULL_IF_ERR_RETURN(new_tensor->grad, "In function nnl2_ad_reverse_derivative_narrow, new_tensor grad is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(origin_tensor->grad, "In function nnl2_ad_reverse_derivative_narrow, origin_tensor grad is NULL");
        
        // Validate dimension bounds
        if(dim >= origin_tensor->grad->rank) {
            NNL2_ERROR("In function nnl2_ad_reverse_derivative_narrow, dimension %u is out of bounds for tensor with %u dimensions", dim, origin_tensor->grad->rank);
			return;
        }
        
        // Validate narrow parameters
        if(start < 0) {
            NNL2_ERROR("In function nnl2_ad_reverse_derivative_narrow, start index %d cannot be negative", start);
			return;
        }
        
        if(len <= 0) {
            NNL2_ERROR("In function nnl2_ad_reverse_derivative_narrow, length %d must be positive", len);
			return;
        }
        
        if((int32_t)(start + len) > origin_tensor->grad->shape[dim]) {
            NNL2_ERROR("In function nnl2_ad_reverse_derivative_narrow, narrow range [%d, %d) exceeds dimension size %u", start, start + len, origin_tensor->grad->shape[dim]);
			return;
        }
    #endif
    
    if (origin_tensor->requires_grad) {
        nnl2_tensor* origin_region = nnl2_narrow(origin_tensor->grad, dim, start, len);
        
        // Add the gradient from the new tensor to the corresponding region
        nnl2_add_inplace(origin_region, new_tensor->grad);
		
        nnl2_free_tensor(origin_region);
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_AD_NARROW_BACKWARD_H **/
