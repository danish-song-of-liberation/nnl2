#ifndef NNL2_ANN_PRINT_H
#define NNL2_ANN_PRINT_H 

// NNL2

/** @file nnl2_ann_print.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Print function
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_print.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Prints an ANN model of any supported type
 *
 ** @param nn 
 * Pointer to the neural network structure to print
 **/
void nnl2_ann_print(void* nn, bool terpri, int depth) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            NNL2_ERROR("In function nnl2_ann_print, void* nn is NULL. nothing to print");
            return;
        }
    #endif
	
    nnl2_nn_ann* ann = (nnl2_nn_ann*)nn;
    
    switch(ann -> nn_type) {
        case nnl2_nn_type_fnn:		   nnl2_nn_fnn_print(nn, terpri);             break;
		case nnl2_nn_type_sigmoid: 	   nnl2_nn_sigmoid_print(nn, terpri);         break;	
		case nnl2_nn_type_tanh: 	   nnl2_nn_tanh_print(nn, terpri);            break;
		case nnl2_nn_type_relu: 	   nnl2_nn_relu_print(terpri); 	              break;	
		case nnl2_nn_type_leaky_relu:  nnl2_nn_leaky_relu_print(nn, terpri);      break;	
		case nnl2_nn_type_sequential:  nnl2_nn_sequential_print(nn, depth + 1);   break;
  
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_print, unknown ann type");
            break;
        }
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
        NNL2_FUNC_EXIT();
    #endif
}

#endif /** NNL2_ANN_PRINT_H **/
