#ifndef NNL2_ANN_MUTATION_H 
#define NNL2_ANN_MUTATION_H 

/** @brief
 * Performs uniform mutation on a neural network layer
 * Creates a new layer where parameters are mutated by adding random values
 * within [-delta, delta] based on mutation rate
 *
 ** @param parent
 * Pointer to the parent neural network layer
 *
 ** @param mutate_rate
 * Probability (0.0 to 1.0) of mutating each element
 *
 ** @param delta
 * Maximum absolute value of mutation to be added to elements
 *
 ** @return void*
 * Pointer to the newly created mutated neural network layer
 *
 ** @retval NULL
 * if memory allocation fails, mutation fails, or layer type is not supported
 *
 ** @note
 * Caller is responsible for freeing the returned layer with nnl2_ann_free()
 */
void* nnl2_nn_mutation_uniform(void* parent, float mutate_rate, float delta) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (!parent) {
            NNL2_ERROR("In nnl2_nn_mutation_uniform, parent layer cannot be NULL");
            return NULL;
        }
        
        if (mutate_rate < 0.0f || mutate_rate > 1.0f) {
            NNL2_ERROR("In nnl2_nn_mutation_uniform, mutate_rate must be between 0.0 and 1.0");
            return NULL;
        }
        
        if (delta < 0.0f) {
            NNL2_ERROR("In nnl2_nn_mutation_uniform, delta must be non-negative");
            return NULL;
        }
    #endif

    nnl2_nn_ann* ann = (nnl2_nn_ann*)parent;
    void* result = NULL;

    switch(ann->nn_type) {
        case nnl2_nn_type_fnn: {
            result = nnl2_nn_fnn_mutation_uniform((nnl2_nn_fnn*)parent, mutate_rate, delta);
            break;
		}
            
        case nnl2_nn_type_rnn_cell: {
            result = nnl2_nn_rnn_cell_mutation_uniform((nnl2_nn_rnn_cell*)parent, mutate_rate, delta);
            break;
		}
		
		case nnl2_nn_type_sequential: {		
            result = nnl2_nn_sequential_mutation_uniform((nnl2_nn_sequential*)parent, mutate_rate, delta);
            break;
        }
			
		case nnl2_nn_type_relu: {		
            result = nnl2_nn_relu_create();
            break;
        }
		
		case nnl2_nn_type_leaky_relu: {
			result = nnl2_nn_leaky_relu_deep_copy((nnl2_nn_leaky_relu*)parent);
			break;
		}
		
		case nnl2_nn_type_sigmoid: {
			result = nnl2_nn_sigmoid_deep_copy((nnl2_nn_sigmoid*)parent);
			break;
		}
				
		case nnl2_nn_type_tanh: {
			result = nnl2_nn_tanh_deep_copy((nnl2_nn_tanh*)parent);
			break;
		}
				
		case nnl2_nn_type_unknown:

        default: {
            NNL2_ERROR("In nnl2_nn_mutation_uniform, unknown or unsupported layer type: %d", ann->nn_type);
            result = NULL;
		}
    }
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result; 
}

#endif /** NNL2_ANN_MUTATION_H **/
