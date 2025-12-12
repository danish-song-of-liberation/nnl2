#ifndef NNL2_ANN_FREE_H
#define NNL2_ANN_FREE_H 

// NNL2

/** @file nnl2_ann_free.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Free function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_free.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Safely frees an ANN model of any supported type
 *
 ** @param nn 
 * Pointer to the neural network structure to be freed
 *
 ** @see nnl2_nn_type
 ** @see nnl2_nn_fnn_free
 **/
void nnl2_ann_free(void* nn) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_ann_free, nn is NULL. nothing to free");
            #endif
			
            return;
        }
    #endif
	
    nnl2_nn_ann* ann = (nnl2_nn_ann*)nn;
    
    switch(ann -> nn_type) {
        case nnl2_nn_type_fnn: {
            nnl2_nn_fnn_free(nn);
            break;
        }
		
		case nnl2_nn_type_rnn_cell: {
			nnl2_nn_rnn_cell_free(nn);
			break;
		}
		
		case nnl2_nn_type_sigmoid: {
			nnl2_nn_sigmoid_free(nn);
			break;
		}
		
		case nnl2_nn_type_tanh: {
			nnl2_nn_tanh_free(nn);
			break;
		}
		
		case nnl2_nn_type_relu: {
			nnl2_nn_relu_free(nn);
			break;
		}
		
		case nnl2_nn_type_leaky_relu: {
			nnl2_nn_leaky_relu_free(nn);
			break;
		}
		
		case nnl2_nn_type_sequential: {
			nnl2_nn_sequential_free(nn);
			break;
		}
        
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_free, unknown ann type");
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_ANN_FREE_H **/
