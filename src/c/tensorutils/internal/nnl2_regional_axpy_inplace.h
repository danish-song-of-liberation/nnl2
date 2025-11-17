#ifndef NNL2_REGIONAL_AXPY_INPLACE_H
#define NNL2_REGIONAL_AXPY_INPLACE_H

/** @brief 
 * Performs in-place AXPY operation (alpha * X + Y) on a tensor region
 * 
 ** @param summand 
 * Pointer to the destination tensor (modified in-place)
 *
 ** @param sumend 
 * Pointer to the source tensor to be scaled and added
 *
 ** @param alpha 
 * Scaling factor for the sumend tensor
 *
 ** @param from 
 * Array of starting indices for each dimension defining the region
 *
 ** @param to 
 * Array of ending indices for each dimension defining the region
 *
 ** @warning
 * Strictly internal function. DO not use it in your code
 */
void nnl2_naive_axpy_inplace_region(nnl2_tensor* summand, nnl2_tensor* sumend, float alpha, int* from, int* to) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand, "Summand tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(sumend, "Sumend tensor is NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(from, "From indices are NULL");
        NNL2_CHECK_NULL_IF_ERR_RETURN(to, "To indices are NULL");
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        NNL2_CHECK_NULL_IF_ERR_RETURN(summand->data, "Summand tensor's data is NULL");
		NNL2_CHECK_NULL_IF_ERR_RETURN(sumend->data, "Sumend tensor's data is NULL");
	#endif
    
    int rank = summand->rank;
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        for (int i = 0; i < rank; i++) {
            if (from[i] < 0 || from[i] >= summand->shape[i] || to[i] < 0 || to[i] >= summand->shape[i] || from[i] > to[i]) {
                NNL2_ERROR("Invalid region bounds for dimension %d: from=%d, to=%d, shape=%d", i, from[i], to[i], summand->shape[i]);
                return;
            }
        }
    #endif
    
    // Calculate region dimensions
    int region_dims[rank];
    size_t total_region_elems = 1;
    for (int i = 0; i < rank; i++) {
        region_dims[i] = to[i] - from[i] + 1;
        total_region_elems *= region_dims[i];
    }
    
    // Check if sumend has compatible shape
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_BASIC
        for (int i = 0; i < rank; i++) {
            if (sumend->shape[i] != region_dims[i]) {
                NNL2_ERROR("Sumend tensor shape doesn't match region dimensions");
                return;
            }
        }
    #endif
    
    // Iterate through region
    int coords[rank];
    for(int i = 0; i < rank; i++) coords[i] = 0;
    
    nnl2_tensor_type dtype_summand = summand->dtype;
    nnl2_tensor_type dtype_sumend = sumend->dtype;
    
    if(dtype_summand == dtype_sumend) {
        switch(dtype_summand) {
            case FLOAT64: {
                nnl2_float64* data_summand = (nnl2_float64*)summand->data;
                nnl2_float64* data_sumend = (nnl2_float64*)sumend->data;
                nnl2_float64 alpha_double = (nnl2_float64)alpha;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    // Calculate index in summand 
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    // Calculate index in sumend 
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    data_summand[idx_summand] += data_sumend[idx_sumend] * alpha_double;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            case FLOAT32: {
                nnl2_float32* data_summand = (nnl2_float32*)summand->data;
                nnl2_float32* data_sumend = (nnl2_float32*)sumend->data;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    data_summand[idx_summand] += data_sumend[idx_sumend] * alpha;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            case INT32: {
                nnl2_int32* data_summand = (nnl2_int32*)summand->data;
                nnl2_int32* data_sumend = (nnl2_int32*)sumend->data;
                nnl2_int32 alpha_int = (nnl2_int32)alpha;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    data_summand[idx_summand] += data_sumend[idx_sumend] * alpha_int;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            default: {
				NNL2_TYPE_ERROR(dtype_summand); 
				return;
			}
        }
    } else {
        size_t sumend_step = get_dtype_size(dtype_sumend);
        char* sumend_data = (char*)sumend->data;
        
        switch(dtype_summand) {
            case FLOAT64: {
                nnl2_float64* data_summand = (nnl2_float64*)summand->data;
                nnl2_float64 alpha_double = (nnl2_float64)alpha;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    void* sumend_elem = sumend_data + idx_sumend * sumend_step;
                    data_summand[idx_summand] += nnl2_convert_to_float64(sumend_elem, dtype_sumend) * alpha_double;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            case FLOAT32: {
                nnl2_float32* data_summand = (nnl2_float32*)summand->data;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    void* sumend_elem = sumend_data + idx_sumend * sumend_step;
                    data_summand[idx_summand] += nnl2_convert_to_float32(sumend_elem, dtype_sumend) * alpha;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            case INT32: {
                nnl2_int32* data_summand = (nnl2_int32*)summand->data;
                nnl2_int32 alpha_int = (nnl2_int32)alpha;
				
                for(size_t i = 0; i < total_region_elems; i++) {
                    size_t idx_summand = 0;
                    for(int j = 0; j < rank; j++)  idx_summand += (from[j] + coords[j]) * summand->strides[j];
                    
                    size_t idx_sumend = 0;
                    for(int j = 0; j < rank; j++)  idx_sumend += coords[j] * sumend->strides[j];
                    
                    void* sumend_elem = sumend_data + idx_sumend * sumend_step;
                    data_summand[idx_summand] += nnl2_convert_to_int32(sumend_elem, dtype_sumend) * alpha_int;
                    
                    for(int j = rank - 1; j >= 0; j--) {
                        coords[j]++;
                        if(coords[j] < region_dims[j]) break;
                        coords[j] = 0;
                    }
                }
				
                break;
            }
            
            default: {
				NNL2_TYPE_ERROR(dtype_summand); 
				return;
			}
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** 
 * @ingroup backend_system
 * @brief Backend implementations for regional AXPY inplace operation
 * @details
 * Array follows the common backend registration pattern for regional AXPY operations.
 * Currently registered backends:
 *  - nnl2_naive: Basic reference implementation
 * 
 * @see nnl2_naive
 */
Implementation axpy_inplace_region_backends[] = {
    REGISTER_BACKEND(nnl2_naive_axpy_inplace_region, nnl2_naive, NAIVE_BACKEND_NAME),
};	

/** @brief Main function **/
axpy_inplace_region_fn nnl2_axpy_inplace_region;

#endif /** NNL2_REGIONAL_AXPY_INPLACE_H **/
