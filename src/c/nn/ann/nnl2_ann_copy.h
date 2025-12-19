#ifndef NNL2_ANN_COPY_H
#define NNL2_ANN_COPY_H 

// NNL2

/** @file nnl2_ann_copy.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Deep Copy function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_copy.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl2.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Creates a deep copy of an ANN model of any supported type
 *
 ** @param src 
 * Pointer to the source neural network structure to be copied
 *
 ** @return
 * A pointer to the newly created deep copy of the neural network
 *
 ** @retval NULL 
 * if memory allocation or tensor copying fails, or if the type is unsupported
 *
 ** @note
 * This function performs a deep copy, meaning all internal tensors
 * are copied, not just referenced
 *
 ** @warning 
 * The caller is responsible for freeing the memory by calling
 * `void nnl2_ann_free(void* nn)` on the returned pointer
 *
 ** @see nnl2_ann_free
 ** @see nnl2_nn_fnn_deep_copy
 **/
void* nnl2_ann_deep_copy(const void* src) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(src == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_ann_deep_copy, src is NULL. nothing to copy");
            #endif
            return NULL;
        }
    #endif
    
    const nnl2_nn_ann* ann = (const nnl2_nn_ann*)src;
    void* result = NULL;
    
    switch(ann->nn_type) {
        case nnl2_nn_type_fnn: {
            result = nnl2_nn_fnn_deep_copy((nnl2_nn_fnn*)src);
            break;
        }
		
        case nnl2_nn_type_rnn_cell: {
            result = nnl2_nn_rnn_cell_deep_copy((nnl2_nn_rnn_cell*)src);
            break;
        }
        
        case nnl2_nn_type_sequential: {
            result = nnl2_nn_sequential_deep_copy((nnl2_nn_sequential*)src);
            break;
        }
		
		case nnl2_nn_type_relu: {		
            result = nnl2_nn_relu_create();
            break;
        }
       
		case nnl2_nn_type_leaky_relu: {
			result = nnl2_nn_leaky_relu_deep_copy((nnl2_nn_leaky_relu*)src);
			break;
		}
		
		case nnl2_nn_type_sigmoid: {
			result = nnl2_nn_sigmoid_deep_copy((nnl2_nn_sigmoid*)src);
			break;
		}
		
		case nnl2_nn_type_tanh: {
			result = nnl2_nn_tanh_deep_copy((nnl2_nn_tanh*)src);
			break;
		}
	   
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_deep_copy, unknown or unsupported ann type: %d", ann->nn_type);
            result = NULL;
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        if(result != NULL) {
            NNL2_INFO("Successfully created deep copy of ANN type: %d", ann->nn_type);
        } else {
            NNL2_INFO("Failed to create deep copy of ANN type: %d", ann->nn_type);
        }
		
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_ANN_COPY_H **/
