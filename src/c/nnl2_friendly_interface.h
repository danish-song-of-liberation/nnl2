#ifndef NNL2_FRIENDLY_INTERFACE_H
#define NNL2_FRIENDLY_INTERFACE_H

// NNL2

/** @file nnl2_friendly_interface.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief nnl2 C User Friendly Interface
 *
 * Contatins user friendly interface and 
 * trivial functions/utilities for the user
 *
 ** Filepath: src/c/nnl2_friendly_interface.h
 ** File: nnl2_friendly_interface.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2		
 **/

///@{ [macros]

/** @brief Friendly interface error return value 
    for size_t functions (0 indicates error) */
#define NNL2_FRIENDLY_SIZE_T_FAILURE 0

/** @brief Friendly interface error return value 
    for int32_t functions (-1 indicates error) */ 
#define NNL2_FRIENDLY_INT32_T_FAILURE -1

///@} [macros]



/** @brief 
 * Calculates the total number of elements in an NNL2 tensor object
 *
 ** @param tensor 
 * Pointer to NNL2 tensor object (nnl2_tensor* or nnl2_ad_tensor*)
 *
 ** @return size_t 
 * The total number of elements in the tensor, or NNL2_FRIENDLY_SIZE_T_FAILURE (0) on error
 *
 ** @example
 * nnl2_tensor* abc = nnl2_ones((int[2]) { 5, 5 }, 2, FLOAT64);
 * printf("%zu\n", nnl2_numel(abc)); // -> 25
 * nnl2_free_tensor(abc);
 *
 ** @see nnl2_tensor
 ** @see nnl2_ad_tensor  
 ** @see NNL2_FATAL 
 ** @see product
 **
 ** @static
 ** @forceinline
 **/
static NNL2_FORCE_INLINE size_t nnl2_numel(void* tensor) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!tensor) {
            NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_numel, NULL pointer passed as tensor parameter");
            return NNL2_FRIENDLY_SIZE_T_FAILURE;
        }
    #endif
	
	nnl2_object_type type = *(nnl2_object_type*)tensor;
	
	switch(type) {
		case nnl2_type_ts: {
            size_t result = product(((nnl2_tensor*)tensor) -> shape, ((nnl2_tensor*)tensor) -> rank);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
		
		case nnl2_type_ad: {
            size_t result = product(((nnl2_ad_tensor*)tensor) -> data -> shape, ((nnl2_ad_tensor*)tensor) -> data -> rank);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
                NNL2_FUNC_EXIT();
            #endif
            
            return result;
        }
		
		case nnl2_type_unknown:
		
		default: {
			NNL2_FATAL("[nnl2 C-User-Interface] An incorrected object (%p) was transferred in nnl2_numel function. nnl2_numel accepts ONLY nnl2_tensor* or nnl2_ad_tensor*", tensor);
			return NNL2_FRIENDLY_SIZE_T_FAILURE;
		}
	}
	
	#if defined(__GNUC__) || defined(__clang__)
        __builtin_unreachable();
    #endif
}

/** @brief 
 * Safely retrieves the shape dimension at specified index from an NNL2 tensor object
 *
 ** @param tensor 
 * Pointer to NNL2 tensor object (nnl2_tensor* or nnl2_ad_tensor*)
 *
 ** @param at 
 * Index of the dimension to retrieve (0-based)
 *
 ** @return int32_t 
 * The size of the dimension at index 'at', or -1 on error
 *
 ** @example
 * nnl2_tensor* abc = nnl2_ones((int[2]) { 5, 5 }, 2, FLOAT64);
 * printf("%d\n", nnl2_get_shape_at(abc, 0)); // -> 5
 * nnl2_free_tensor(abc);
 *
 ** @see nnl2_tensor
 ** @see nnl2_ad_tensor  
 ** @see NNL2_FATAL 
 **
 ** @static
 ** @forceinline
 **/
static NNL2_FORCE_INLINE int32_t nnl2_get_shape_at(void* tensor, int64_t at) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(!tensor) {
			NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_shape_at, NULL pointer passed as tensor parameter");
			return NNL2_FRIENDLY_INT32_T_FAILURE;
		}
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
		if(at > INT_MAX) {
			NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_shape_at, integer overflow detected in index 'at'. Value %lld exceeds maximum 32-bit integer value %d", 
					   (long long)at, INT_MAX);
					   
			return NNL2_FRIENDLY_INT32_T_FAILURE;
		}
		
		if(at < INT_MIN) {
			NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_shape_at, integer underflow detected in index 'at'. Value %lld is less than minimum 32-bit integer value %d", 
					   (long long)at, INT_MIN);
					   
			return NNL2_FRIENDLY_INT32_T_FAILURE;
		}
	#endif
    
    nnl2_object_type type = *(nnl2_object_type*)tensor;
    
    switch(type) {
        case nnl2_type_ts: {
            if((at >= (((nnl2_tensor*)tensor) -> rank)) || (at < 0)) {
                NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_shape_at, index 'at' (%lld) is out of bounds for tensor with rank %d. Index must be in the range [0, %d]",
                            (long long)at, ((nnl2_tensor*)tensor)->rank, ((nnl2_tensor*)tensor)->rank - 1);
							
                return NNL2_FRIENDLY_INT32_T_FAILURE;
				
            } else {
				// Direct memory access to shape array
				int32_t result = *((((nnl2_tensor*)tensor) -> shape) + at);
				
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
					NNL2_FUNC_EXIT();
				#endif
	
                return result;
            }
        }
        
        case nnl2_type_ad: {
            if((at >= (((nnl2_ad_tensor*)tensor) -> data -> rank)) || (at < 0)) {
                NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_shape_at, index 'at' (%lld) is out of bounds for autodiff tensor with rank %d. Index must be in the range [0, %d]",
                            (long long)at, ((nnl2_ad_tensor*)tensor)->data->rank, ((nnl2_ad_tensor*)tensor)->data->rank - 1);
							
                return NNL2_FRIENDLY_INT32_T_FAILURE;
				
            } else {
				// Direct memory access to shape array
                int32_t result = *((((nnl2_ad_tensor*)tensor) -> data -> shape) + at);   
				
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
					NNL2_FUNC_EXIT();
				#endif

				return result;
            }
        }
        
        case nnl2_type_unknown:
        
        default: {
            NNL2_FATAL("[nnl2 C-User-Interface] An incorrected object (%p) was transferred in nnl2_shape_at function. nnl2_shape_at accepts ONLY nnl2_tensor* or nnl2_ad_tensor*", tensor);
            return NNL2_FRIENDLY_INT32_T_FAILURE;
        }
    }
	
	#if defined(__GNUC__) || defined(__clang__)
		__builtin_unreachable();
	#endif
}

/** @brief 
 * Safely retrieves the stride at specified index from an NNL2 tensor object
 *
 ** @param tensor 
 * Pointer to NNL2 tensor object (nnl2_tensor* or nnl2_ad_tensor*)
 *
 ** @param at 
 * Index of the stride to retrieve (0-based)
 *
 ** @return int32_t 
 * The stride value at index 'at', or -1 on error
 *
 ** @example
 * nnl2_tensor* abc = nnl2_ones((int[2]) { 5, 5 }, 2, FLOAT64);
 * printf("%d\n", nnl2_get_stride_at(abc, 0)); // -> 5 (for row-major)
 * nnl2_free_tensor(abc);
 *
 ** @see nnl2_tensor
 ** @see nnl2_ad_tensor  
 ** @see NNL2_FATAL 
 **
 ** @static
 ** @forceinline
 **/
static NNL2_FORCE_INLINE int32_t nnl2_get_stride_at(void* tensor, int64_t at) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!tensor) {
            NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_stride_at, NULL pointer passed as tensor parameter");
            return NNL2_FRIENDLY_INT32_T_FAILURE;
        }
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MAX
        if(at > INT_MAX) {
            NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_stride_at, integer overflow detected in index 'at'. Value %lld exceeds maximum 32-bit integer value %d", 
                       (long long)at, INT_MAX);
                       
            return NNL2_FRIENDLY_INT32_T_FAILURE;
        }
        
        if(at < INT_MIN) {
            NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_stride_at, integer underflow detected in index 'at'. Value %lld is less than minimum 32-bit integer value %d", 
                       (long long)at, INT_MIN);
                       
            return NNL2_FRIENDLY_INT32_T_FAILURE;
        }
    #endif
    
    nnl2_object_type type = *(nnl2_object_type*)tensor;
    
    switch(type) {
        case nnl2_type_ts: {
            if((at >= (((nnl2_tensor*)tensor) -> rank)) || (at < 0)) {
                NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_stride_at, index 'at' (%lld) is out of bounds for tensor with rank %d. Index must be in the range [0, %d]",
                            (long long)at, ((nnl2_tensor*)tensor)->rank, ((nnl2_tensor*)tensor)->rank - 1);
                            
                return NNL2_FRIENDLY_INT32_T_FAILURE;
                
            } else {
                // Direct memory access to strides array
                int32_t result = *((((nnl2_tensor*)tensor) -> strides) + at);
                
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
                    NNL2_FUNC_EXIT();
                #endif
    
                return result;
            }
        }
        
        case nnl2_type_ad: {
            if((at >= (((nnl2_ad_tensor*)tensor) -> data -> rank)) || (at < 0)) {
                NNL2_FATAL("[nnl2 C-User-Interface] In function nnl2_get_stride_at, index 'at' (%lld) is out of bounds for autodiff tensor with rank %d. Index must be in the range [0, %d]",
                            (long long)at, ((nnl2_ad_tensor*)tensor)->data->rank, ((nnl2_ad_tensor*)tensor)->data->rank - 1);
                            
                return NNL2_FRIENDLY_INT32_T_FAILURE;
                
            } else {
                // Direct memory access to strides array
                int32_t result = *((((nnl2_ad_tensor*)tensor) -> data -> strides) + at);   
                
                #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
                    NNL2_FUNC_EXIT();
                #endif

                return result;
            }
        }
        
        case nnl2_type_unknown:
        
        default: {
            NNL2_FATAL("[nnl2 C-User-Interface] An incorrected object (%p) was transferred in nnl2_get_stride_at function. nnl2_get_stride_at accepts ONLY nnl2_tensor* or nnl2_ad_tensor*", tensor);
            return NNL2_FRIENDLY_INT32_T_FAILURE;
        }
    }
    
    #if defined(__GNUC__) || defined(__clang__)
        __builtin_unreachable();
    #endif
}

#endif /** NNL2_FRIENDLY_INTERFACE_H **/
