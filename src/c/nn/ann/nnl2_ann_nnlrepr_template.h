#ifndef NNL2_ANN_NNLREPR_TEMPLATE_H
#define NNL2_ANN_NNLREPR_TEMPLATE_H
	
// NNL2

/** @file nnl2_ann_free.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN generic function for nnlrepr encoding
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_nnlrepr_template.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief 
 * Encodes by nnlrepr format passed neural network
 *
 ** @param nn 
 * Input neural network
 */
nnl2_nnlrepr_template* nnl2_ann_nnlrepr_template(void* nn) {
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_ann_nnlrepr_template, nn is NULL");
            #endif
			
            return NULL;
        }
    #endif
	
    nnl2_nn_ann* ann = (nnl2_nn_ann*)nn;
    
    switch(ann -> nn_type) {
        case nnl2_nn_type_fnn:			return nnl2_nn_fnn_nnlrepr_template((nnl2_nn_fnn*)nn);
		case nnl2_nn_type_rnn_cell:		return nnl2_nn_rnn_cell_nnlrepr_template((nnl2_nn_rnn_cell*)nn);
		case nnl2_nn_type_sigmoid: 	    return nnl2_nn_sigmoid_nnlrepr_template(nn);
		case nnl2_nn_type_tanh: 		return nnl2_nn_tanh_nnlrepr_template(nn);
		case nnl2_nn_type_relu: 	    return nnl2_nn_relu_nnlrepr_template();
		case nnl2_nn_type_leaky_relu:   return nnl2_nn_leaky_relu_nnlrepr_template(nn);
		case nnl2_nn_type_sequential:   return nnl2_nn_sequential_nnlrepr_template((nnl2_nn_sequential*)nn);
		
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In function nnl2_ann_nnlrepr_template, unknown ann type");
            return NULL;
        }
    }
	
    #if defined(__GNUC__) || defined(__clang__)
		__builtin_unreachable();
	#endif
}

#endif /** NNL2_ANN_NNLREPR_TEMPLATE_H **/
