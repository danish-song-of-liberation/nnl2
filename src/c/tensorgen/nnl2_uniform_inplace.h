#ifndef NNL2_UNIFORM_INPLACE_H
#define NNL2_UNIFORM_INPLACE_H

/** @file nnl2_uniform_inplace.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains definition of functions that fills tensor with random values in-place
 **/
 
/** @brief
 * Fills the given tensor with random values from the specified range (in-place)
 *
 ** @param tensor 
 * Tensor to fill with random values
 *
 ** @param from 
 * Pointer to the lower bound of the random range
 *
 ** @param to 
 * Pointer to the upper bound of the random range
 *
 ** @exception NNL2Error [nnl2_safety_mode_min+]
 * If tensor is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mode_moderate+]
 * If from is NULL
 *
 ** @exception NNL2Error [nnl2_safety_mode_moderate+]
 * If to is NULL
 *
 ** @exception NNL2Error 
 * If passed tensor with unknown type
 *
 ** @see nnl2_product
 **/ 
void nnl2_naive_uniform_inplace(nnl2_tensor* tensor, void* from, void* to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
		NNL2_FUNC_ENTER(); 
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_naive_uniform_inplace, passed tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(from, "In function nnl2_naive_uniform_inplace, from is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(to, "In function nnl2_naive_uniform_inplace, to is NULL");
	#endif
	
	size_t total_elems = nnl2_product(tensor->shape, tensor->rank);
	if(total_elems == 0) return; // If zero elems then early return
	
	// Filling with random values
	switch(tensor->dtype) {
		case FLOAT64: {
			nnl2_float64 from_cast = *((nnl2_float64*)from);
			nnl2_float64 to_cast = *((nnl2_float64*)to);
			nnl2_float64* data = (nnl2_float64*)tensor->data;
			for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((double)rand() / RAND_MAX);
			break;
		}
		
		case FLOAT32: {
            nnl2_float32 from_cast = *((nnl2_float32*)from);
            nnl2_float32 to_cast = *((nnl2_float32*)to);
            nnl2_float32* data = (nnl2_float32*)tensor->data;
            for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + (to_cast - from_cast) * ((float)rand() / RAND_MAX);
            break;
        }
		
		case INT64: {
            nnl2_int64 from_cast = *((nnl2_int64*)from);
            nnl2_int64 to_cast = *((nnl2_int64*)to);
            nnl2_int64* data = (nnl2_int64*)tensor->data;
            nnl2_int64 range = to_cast - from_cast + 1;
            
            for(size_t i = 0; i < total_elems; i++) {
                double rand_ratio = (double)rand() / RAND_MAX;
                data[i] = from_cast + (nnl2_int64)(rand_ratio * range);
            }
			
            break;
        }

        case INT32: {
            nnl2_int32 from_cast = *((nnl2_int32*)from);
            nnl2_int32 to_cast = *((nnl2_int32*)to);
            nnl2_int32* data = (nnl2_int32*)tensor->data;
            for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + rand() % (to_cast - from_cast + 1);
            break;
        }
		
		case FLOAT128: {
			nnl2_float128 from_cast = (nnl2_float128)(*(nnl2_float64*)from);
			nnl2_float128 to_cast = (nnl2_float128)(*(nnl2_float64*)to);
			nnl2_float128* data = (nnl2_float128*)tensor->data;
			
			for(size_t i = 0; i < total_elems; i++) {
				nnl2_float128 rand_ratio = (nnl2_float128)rand() / RAND_MAX;
				data[i] = from_cast + (to_cast - from_cast) * rand_ratio;
			}
			break;
		}

		case INT8: {
			nnl2_int8 from_cast = *((nnl2_int8*)from);
			nnl2_int8 to_cast = *((nnl2_int8*)to);
			nnl2_int8* data = (nnl2_int8*)tensor->data;
			nnl2_int8 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = from_cast + rand() % range;
			}
			break;
		}

		case INT16: {
			nnl2_int16 from_cast = *((nnl2_int16*)from);
			nnl2_int16 to_cast = *((nnl2_int16*)to);
			nnl2_int16* data = (nnl2_int16*)tensor->data;
			nnl2_int16 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = from_cast + rand() % range;
			}
			break;
		}

		case UINT8: {
			nnl2_uint8 from_cast = *((nnl2_uint8*)from);
			nnl2_uint8 to_cast = *((nnl2_uint8*)to);
			nnl2_uint8* data = (nnl2_uint8*)tensor->data;
			nnl2_uint8 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = from_cast + rand() % range;
			}
			break;
		}

		case UINT16: {
			nnl2_uint16 from_cast = *((nnl2_uint16*)from);
			nnl2_uint16 to_cast = *((nnl2_uint16*)to);
			nnl2_uint16* data = (nnl2_uint16*)tensor->data;
			nnl2_uint16 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = from_cast + rand() % range;
			}
			break;
		}

		case UINT32: {
			nnl2_uint32 from_cast = *((nnl2_uint32*)from);
			nnl2_uint32 to_cast = *((nnl2_uint32*)to);
			nnl2_uint32* data = (nnl2_uint32*)tensor->data;
			nnl2_uint32 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				data[i] = from_cast + rand() % range;
			}
			break;
		}

		case UINT64: {
			nnl2_uint64 from_cast = *((nnl2_uint64*)from);
			nnl2_uint64 to_cast = *((nnl2_uint64*)to);
			nnl2_uint64* data = (nnl2_uint64*)tensor->data;
			nnl2_uint64 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				double rand_ratio = (double)rand() / RAND_MAX;
				data[i] = from_cast + (nnl2_uint64)(rand_ratio * range);
			}
			break;
		}

		case INT128: {
			nnl2_int128 from_cast = (nnl2_int128)(*(nnl2_float64*)from);
			nnl2_int128 to_cast = (nnl2_int128)(*(nnl2_float64*)to);
			nnl2_int128* data = (nnl2_int128*)tensor->data;
			nnl2_int128 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				double rand_ratio = (double)rand() / RAND_MAX;
				data[i] = from_cast + (nnl2_int128)(rand_ratio * range);
			}
			break;
		}

		case UINT128: {
			nnl2_uint128 from_cast = (nnl2_uint128)(*(nnl2_float64*)from);
			nnl2_uint128 to_cast = (nnl2_uint128)(*(nnl2_float64*)to);
			nnl2_uint128* data = (nnl2_uint128*)tensor->data;
			nnl2_uint128 range = to_cast - from_cast + 1;
			
			for(size_t i = 0; i < total_elems; i++) {
				double rand_ratio = (double)rand() / RAND_MAX;
				data[i] = from_cast + (nnl2_uint128)(rand_ratio * range);
			}
			break;
		}

		case BOOL: {
			nnl2_bool from_cast = *((nnl2_bool*)from);
			nnl2_bool to_cast = *((nnl2_bool*)to);
			nnl2_bool* data = (nnl2_bool*)tensor->data;
			
			if (from_cast == false && to_cast == true) {
				for(size_t i = 0; i < total_elems; i++) {
					data[i] = (rand() % 2) ? true : false;
				}
			} else {
				for(size_t i = 0; i < total_elems; i++) {
					data[i] = from_cast;
				}
			}
			break;
		}
		
		default: {
			NNL2_TYPE_ERROR(tensor->dtype); 
			return;
		}
	}
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
		NNL2_FUNC_EXIT(); 
	#endif
}

/**
 * @ingroup backend_system
 * @brief Backend implementations for uniform_inplace operation
 * @details
 * Array follows the common backend registration pattern for random number 
 * generation operations (in-place version). Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for random number generation (in-place)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_uniform_inplace
 */
Implementation uniform_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_uniform_inplace, nnl2_naive, NAIVE_BACKEND_NAME), // DO NOT OPTIMIZE
};

/**
 * @brief Function pointer for uniform_inplace operation
 * @ingroup backend_system
 */
uniforminplacefn uniform_inplace;

/** 
 * @brief Makes the uniform_inplace backend current
 * @ingroup backend_system
 * @see MAKE_CURRENT_BACKEND
 */
MAKE_CURRENT_BACKEND(uniform_inplace);

/** 
 * @brief Sets the backend for uniform_inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for uniform_inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_uniform_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(uniform_inplace_backends, uniform_inplace, backend_name, CURRENT_BACKEND(uniform_inplace));
}

/** 
 * @brief Gets the name of the active backend for uniform_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_uniform_inplace_backend() {
    return CURRENT_BACKEND(uniform_inplace);
}

/**
 * @brief Function declaration for getting all available uniform_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(uniform_inplace);

/**
 * @brief Function declaration for getting the number of available uniform_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(uniform_inplace);

#endif /** NNL2_UNIFORM_INPLACE_H **/
