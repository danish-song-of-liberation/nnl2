#ifndef NNL2_ANN_NUM_PARAMETERS_H
#define NNL2_ANN_NUM_PARAMETERS_H 

// NNL2

/** @file nnl2_ann_num_parameters.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Number of Parameters function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_num_parameters.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Returns the number of parameters in an ANN model
 *
 ** @param nn 
 * Pointer to the neural network structure
 *
 ** @return size_t
 * Number of parameters in the model, or 0 on error/unknown type
 *
 ** @see nnl2_nn_type
 ** @see nnl2_nn_fnn_get_num_parameters
 **/
size_t nnl2_ann_num_parameters(void* nn) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_ann_num_parameters, nn is NULL. returning 0");
            #endif
			
            return 0;
        }
    #endif
    
    nnl2_nn_type* nn_type = (nnl2_nn_type*)nn;
    
    switch(*nn_type) {
        case nnl2_nn_type_fnn: {
            size_t num_params = nnl2_nn_fnn_get_num_parameters(nn);
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return num_params;	
        }
        
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_num_parameters, unknown ann type");
            
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_FUNC_EXIT();
            #endif
            
            return 0;
        }
    }
}

#endif /** NNL2_ANN_NUM_PARAMETERS_H **/
