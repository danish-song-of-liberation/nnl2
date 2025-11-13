#ifndef NNL2_AD_NCAST_H
#define NNL2_AD_NCAST_H

// NNL2

/** @file nnl2_ad_ncast.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains function to cast ad-tensor type
 **/

/** @brief
 * Casts an automatic differentiation tensor to a different data type
 *
 ** @param ad_tensor
 * Pointer to the input automatic differentiation tensor to be cast
 *
 ** @param dtype
 * Target data type for the type casting operation
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If ad_tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If result is NULL
 *
 ** @return nnl2_ad_tensor*
 * Pointer to a new automatic differentiation tensor with the specified data type,
 * or NULL if the operation fails
 *
 ** @see nnl2_ad_copy
 **/
nnl2_ad_tensor* nnl2_ad_ncast(nnl2_ad_tensor* ad_tensor, nnl2_tensor_type dtype) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_ENTER();
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(ad_tensor, "In function nnl2_ad_ncast, ad_tensor is NULL", NULL);
	#endif
	
	// Type-casting
	nnl2_ad_tensor* result = nnl2_ad_copy(ad_tensor, dtype);
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(result, "In function nnl2_ad_ncast, result is NULL", NULL);
	#endif
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
		NNL2_FUNC_EXIT();
	#endif
	
	return result;
}

#endif /** NNL2_AD_NCAST_H **/
