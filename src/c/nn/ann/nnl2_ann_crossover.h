#ifndef NNL2_NN_CROSSOVER_UNIFORM_H
#define NNL2_NN_CROSSOVER_UNIFORM_H

/** @file nnl2_nn_crossover_uniform.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains Neural Network Crossover Uniform function
 **
 ** Filepath: src/c/nn/ann/nnl2_nn_crossover_uniform.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

/** @brief
 * Performs uniform crossover between two neural network layers of the same type
 *
 ** @param parent_x
 * Pointer to the first parent neural network layer
 *
 ** @param parent_y
 * Pointer to the second parent neural network layer
 *
 ** @param crossover_rate
 * Probability (0.0 to 1.0) of selecting elements from parent_x
 *
 ** @return void*
 * Pointer to the newly created child neural network layer
 */
void* nnl2_nn_crossover_uniform(void* parent_x, void* parent_y, float crossover_rate) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (!parent_x || !parent_y) {
            NNL2_ERROR("In nnl2_nn_crossover_uniform, parent layers cannot be NULL");
            return NULL;
        }
        
        if (crossover_rate < 0.0f || crossover_rate > 1.0f) {
            NNL2_ERROR("In nnl2_nn_crossover_uniform, crossover_rate must be between 0.0 and 1.0");
            return NULL;
        }
    #endif

    nnl2_nn_ann* ann_x = (nnl2_nn_ann*)parent_x;
    nnl2_nn_ann* ann_y = (nnl2_nn_ann*)parent_y;
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (ann_x->nn_type != ann_y->nn_type) {
            NNL2_ERROR("In nnl2_nn_crossover_uniform, parent layers must be of the same type");
            return NULL;
        }
    #endif

    void* result = NULL;

    switch (ann_x->nn_type) {
        case nnl2_nn_type_fnn: {
            result = nnl2_nn_fnn_crossover_uniform((nnl2_nn_fnn*)parent_x, (nnl2_nn_fnn*)parent_y, crossover_rate);
            break;
		}
            
        case nnl2_nn_type_rnn_cell: {
            result = nnl2_nn_rnn_cell_crossover_uniform((nnl2_nn_rnn_cell*)parent_x, (nnl2_nn_rnn_cell*)parent_y, crossover_rate);
            break;
		}
		
		case nnl2_nn_type_sequential: {
            result = nnl2_nn_sequential_crossover_uniform((nnl2_nn_sequential*)parent_x, (nnl2_nn_sequential*)parent_y, crossover_rate);
            break;
        }
		
        case nnl2_nn_type_relu: {		
            result = nnl2_nn_relu_create();
            break;
        }
		
		case nnl2_nn_type_leaky_relu: {
			result = nnl2_nn_leaky_relu_deep_copy((nnl2_nn_leaky_relu*)parent_x);
			break;
		}
		
		case nnl2_nn_type_sigmoid: {
			result = nnl2_nn_sigmoid_deep_copy((nnl2_nn_sigmoid*)parent_x);
			break;
		}
		
		case nnl2_nn_type_tanh: {
			result = nnl2_nn_tanh_deep_copy((nnl2_nn_tanh*)parent_x);
			break;
		}
		
        case nnl2_nn_type_unknown:
		
        default: {
            NNL2_ERROR("In nnl2_nn_crossover_uniform, unknown or unsupported layer type: %d", ann_x->nn_type);
            result = NULL;
            break;
		}
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result; 
}

#endif /** NNL2_NN_CROSSOVER_UNIFORM_H **/
