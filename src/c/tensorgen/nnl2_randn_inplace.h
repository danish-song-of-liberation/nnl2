#ifndef NNL2_RANDN_INPLACE_H
#define NNL2_RANDN_INPLACE_H

/** @file nnl2_randn_inplace.h
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
 ** @see product
 **/ 
void nnl2_naive_randn_inplace(nnl2_tensor* tensor, void* from, void* to) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE 
		NNL2_FUNC_ENTER(); 
	#endif
	
	// Safety checks
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		NNL2_CHECK_NULL_IF_ERR_RETURN(tensor, "In function nnl2_naive_randn_inplace, passed tensor is NULL");
	#endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
		NNL2_CHECK_NULL_IF_ERR_RETURN(from, "In function nnl2_naive_randn_inplace, from is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(to, "In function nnl2_naive_randn_inplace, to is NULL");
	#endif
	
	size_t total_elems = product(tensor->shape, tensor->rank);
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

        case INT32: {
            nnl2_int32 from_cast = *((nnl2_int32*)from);
            nnl2_int32 to_cast = *((nnl2_int32*)to);
            nnl2_int32* data = (nnl2_int32*)tensor->data;
            for(size_t i = 0; i < total_elems; i++) data[i] = from_cast + rand() % (to_cast - from_cast + 1);
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
 * @brief Backend implementations for randn_inplace operation
 * @details
 * Array follows the common backend registration pattern for random number 
 * generation operations (in-place version). Currently registered backends:
 *  - nnl2_naive: Basic reference implementation for random number generation (in-place)
 * 
 * @see nnl2_naive
 * @see nnl2_naive_randn_inplace
 */
Implementation randn_inplace_backends[] = {
    REGISTER_BACKEND(nnl2_naive_randn_inplace, nnl2_naive, NAIVE_BACKEND_NAME), // DO NOT OPTIMIZE
};

/**
 * @brief Function pointer for randn_inplace operation
 * @ingroup backend_system
 */
randninplacefn randn_inplace;

/** 
 * @brief Makes the randn_inplace backend current
 * @ingroup backend_system
 * @see make_current_backend
 */
make_current_backend(randn_inplace);

/** 
 * @brief Sets the backend for randn_inplace operation
 * @ingroup backend_system
 * @param backend_name Name of the backend to activate for randn_inplace
 * @see ESET_BACKEND_BY_NAME
 */
void set_randn_inplace_backend(const char* backend_name) {
    ESET_BACKEND_BY_NAME(randn_inplace_backends, randn_inplace, backend_name, current_backend(randn_inplace));
}

/** 
 * @brief Gets the name of the active backend for randn_inplace operation
 * @ingroup backend_system
 * @return Name of the current backend as constant string
 */
const char* get_randn_inplace_backend() {
    return current_backend(randn_inplace);
}

/**
 * @brief Function declaration for getting all available randn_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_BACKENDS_FUNCTION
 */
DEFINE_GET_BACKENDS_FUNCTION(randn_inplace);

/**
 * @brief Function declaration for getting the number of available randn_inplace backends
 * @ingroup backend_system
 * @see DEFINE_GET_NUMS_BACKENDS_FUNCTION
 */
DEFINE_GET_NUMS_BACKENDS_FUNCTION(randn_inplace);

#endif /** NNL2_RANDN_INPLACE_H **/
