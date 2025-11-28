#ifndef NNL2_ANN_PARAMETERS_FREE_H
#define NNL2_ANN_PARAMETERS_FREE_H 

// NNL2

/** @file nnl2_ann_parameters_free.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Parameters Free function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_parameters_free.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Safely frees an array of ANN parameters (tensors)
 *
 ** @param parameters 
 * Pointer to array of nnl2_ad_tensor pointers to be freed
 *
 ** @see nnl2_ad_tensor
 **/
void nnl2_ann_free_parameters(nnl2_ad_tensor** parameters) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(parameters == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
                NNL2_DEBUG("In function nnl2_ann_free_parameters, parameters is NULL. nothing to free");
            #endif
			
            return;
        }
    #endif
    
    free(parameters);
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_ANN_PARAMETERS_FREE_H **/
