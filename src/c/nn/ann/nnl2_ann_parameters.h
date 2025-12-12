#ifndef NNL2_ANN_PARAMETERS_H
#define NNL2_ANN_PARAMETERS_H 

// NNL2

/** @file nnl2_ann_parameters.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Parameters retrieval function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_parameters.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Retrieves all parameters from an ANN model
 *
 ** @param nn 
 * Pointer to the neural network structure. Must be a valid ANN type.
 * If NULL, function returns NULL.
 *
 ** @return nnl2_ad_tensor**
 * Array of tensors representing the model's parameters, or NULL on error/unknown type
 *
 ** @see nnl2_nn_type
 ** @see nnl2_nn_fnn_get_parameters
 ** @see nnl2_nn_sequential_get_parameters
 **/
nnl2_ad_tensor** nnl2_ann_parameters(void* nn) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_ann_parameters, nn is NULL. returning NULL");
            #endif
			
            return NULL;
        }
    #endif
    
    nnl2_nn_type* nn_type = (nnl2_nn_type*)nn;
    
    switch(*nn_type) {
        case nnl2_nn_type_fnn: {
            nnl2_ad_tensor** parameters = nnl2_nn_fnn_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_unirnn_cell: {
            nnl2_ad_tensor** parameters = nnl2_nn_unirnn_cell_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_sigmoid: {
            nnl2_ad_tensor** parameters = nnl2_nn_sigmoid_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_tanh: {
            nnl2_ad_tensor** parameters = nnl2_nn_tanh_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_relu: {
            nnl2_ad_tensor** parameters = nnl2_nn_relu_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_leaky_relu: {
            nnl2_ad_tensor** parameters = nnl2_nn_leaky_relu_get_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return parameters;	
        }
		
		case nnl2_nn_type_sequential: {
			nnl2_ad_tensor** parameters = nnl2_nn_sequential_get_parameters(nn);
			
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
			
			return parameters;
		}
        
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_parameters, unknown ann type");
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return NULL;
        }
    }
}

#endif /** NNL2_ANN_PARAMETERS_H **/
