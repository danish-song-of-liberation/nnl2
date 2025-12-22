#ifndef NNL2_NN_RNN_BACKEND_H
#define NNL2_NN_RNN_BACKEND_H

typedef struct nnl2_nn_rnn_struct {
    nnl2_nn_ann metadata;
    nnl2_nn_rnn_cell** cells;
    int hidden_size;
	int num_layers;
	
	nnl2_ad_tensor* (*forward)(
		struct nnl2_nn_rnn_struct*,
		nnl2_ad_tensor*
    );
} nnl2_nn_rnn;

nnl2_ad_tensor* nnl2_nn_rnn_forward_with_bias(nnl2_nn_rnn* rnn, nnl2_ad_tensor* inputs);

nnl2_nn_rnn* nnl2_nn_rnn_create(int input_size, int hidden_size, bool bias, int num_layers, nnl2_tensor_type dtype, nnl2_nn_init_type init_type) {
	nnl2_nn_rnn* rnn = malloc(sizeof(nnl2_nn_rnn));
	
	rnn->metadata.nn_type = nnl2_nn_type_rnn;
	rnn->metadata.nn_magic = NNL2_NN_MAGIC;
	
	rnn->metadata.use_bias = bias;
	
	rnn->hidden_size = hidden_size;
	rnn->num_layers = num_layers;
	
	rnn->cells = malloc(sizeof(nnl2_nn_rnn_cell*) * num_layers);
	
	for(int i = 0; i < num_layers; ++i) {
        int layer_input_size = (i == 0) ? input_size : hidden_size;
		rnn->cells[i] = nnl2_nn_rnn_cell_create(layer_input_size, hidden_size, bias, dtype, init_type);
	}
	
	rnn->forward = nnl2_nn_rnn_forward_with_bias;
	
	return rnn;
}

nnl2_ad_tensor* nnl2_nn_rnn_forward_with_bias(nnl2_nn_rnn* rnn, nnl2_ad_tensor* inputs) {
   // NNL2_DEBUG("[RNN] Forward start: T=%d, hidden_size=%d, num_layers=%d",
     //          inputs->data->shape[0], rnn->hidden_size, rnn->num_layers);

    int T = inputs->data->shape[0];
    nnl2_ad_tensor* layer_input = inputs;

    for(int l = 0; l < rnn->num_layers; ++l) {
        //NNL2_DEBUG("[RNN] Layer %d/%d", l+1, rnn->num_layers);

        nnl2_ad_tensor* h = nnl2_ad_zeros((int[]){1, rnn->hidden_size}, 2,
                                         layer_input->data->dtype, true,
                                         "(Created by nnl2.hli.nn automatically) RNN zeros");
        if(!h) { NNL2_ERROR("[RNN] Failed to allocate hidden state"); return NULL; }

        nnl2_ad_tensor* outputs = nnl2_ad_empty((int[]){T, 1, rnn->hidden_size}, 3,
                                               layer_input->data->dtype, true, NULL);
        if(!outputs) { NNL2_ERROR("[RNN] Failed to allocate outputs"); nnl2_free_ad_tensor(h); return NULL; }

        for(int t = 0; t < T; ++t) {
            //NNL2_DEBUG("[RNN] Layer %d timestep %d", l+1, t);

            nnl2_ad_tensor* x_t = nnl2_ad_tensor_timestep_view(layer_input, t);
            if(!x_t) { NNL2_ERROR("[RNN] Failed to create timestep view at t=%d", t); nnl2_free_ad_tensor(outputs); nnl2_free_ad_tensor(h); return NULL; }

            //NNL2_DEBUG("[RNN] x_t shape: [%d x %d]", x_t->data->shape[0], x_t->data->shape[1]);

            h = nnl2_nn_rnn_cell_forward_with_bias(rnn->cells[l], x_t, h);
            if(!h) { NNL2_ERROR("[RNN] RNN cell forward returned NULL at t=%d", t); nnl2_free_ad_tensor(outputs); return NULL; }

            nnl2_ad_assign_row(outputs, t, h, true);
        }

        layer_input = outputs;
    }

    //NNL2_DEBUG("[RNN] Forward finished");
    return layer_input;
}

/** @brief 
 * Destroys a Recurrent Neural Network (RNN) layer and releases its memory
 *
 ** @param rnn 
 * Pointer to the RNN layer to be destroyed
 *
 ** @warning 
 * This function also frees all RNN cells within the layer
 *
 ** @see nnl2_nn_rnn_create
 **/
void nnl2_nn_rnn_free(nnl2_nn_rnn* rnn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!rnn) {
            NNL2_ERROR("In function nnl2_nn_rnn_free, nnl2_nn_rnn* rnn is NULL");
            return;
        }
    #endif
    
    // Free all RNN cells
    if(rnn->cells) {
        for(int i = 0; i < rnn->num_layers; ++i) {
            if(rnn->cells[i]) {
                nnl2_nn_rnn_cell_free(rnn->cells[i]);
                rnn->cells[i] = NULL;
            }
        }
        
        free(rnn->cells);
        rnn->cells = NULL;
    }
    
    rnn->metadata.nn_magic = 0;
    rnn->metadata.nn_type = nnl2_nn_type_unknown;
    
    free(rnn);
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves all trainable parameters from a Recurrent Neural Network (RNN) layer
 *
 ** @param rnn 
 * Pointer to the RNN layer
 *
 ** @return 
 * Dynamically allocated array of parameter tensor pointers
 *
 ** @retval 
 * NULL if memory allocation fails or if rnn is invalid
 *
 ** @note 
 * The caller is responsible for freeing the returned array with 
 * `void nnl2_ann_free_parameters(nnl2_ad_tensor** parameters)`
 *
 ** @see nnl2_ann_free_parameters
 ** @see nnl2_nn_rnn_get_num_parameters
 **/
nnl2_ad_tensor** nnl2_nn_rnn_get_parameters(nnl2_nn_rnn* rnn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!rnn) {
            NNL2_ERROR("In function nnl2_nn_rnn_get_parameters, nnl2_nn_rnn* rnn is NULL");
            return NULL;
        }
        
        if(rnn->metadata.nn_magic != NNL2_NN_MAGIC) {
            NNL2_ERROR("Invalid RNN structure (bad magic number)");
            return NULL;
        }
    #endif
    
    size_t total_params = 0;
    for(int i = 0; i < rnn->num_layers; ++i) {
        if(rnn->cells[i]) {
            total_params += nnl2_nn_rnn_cell_get_num_parameters(rnn->cells[i]);
        }
    }
    
    if(total_params == 0) {
        NNL2_WARN("RNN has no parameters or cells are not properly initialized");
        return NULL;
    }
    
    nnl2_ad_tensor** params = malloc(sizeof(nnl2_ad_tensor*) * total_params);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!params) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
    
    size_t param_idx = 0;
    for(int layer = 0; layer < rnn->num_layers; ++layer) {
        if(!rnn->cells[layer]) {
            NNL2_WARN("RNN cell at layer %d is NULL", layer);
            continue;
        }
        
        nnl2_ad_tensor** cell_params = nnl2_nn_rnn_cell_get_parameters(rnn->cells[layer]);
        if(!cell_params) {
            NNL2_WARN("Failed to get parameters for RNN cell at layer %d", layer);
            continue;
        }
        
        size_t cell_param_count = nnl2_nn_rnn_cell_get_num_parameters(rnn->cells[layer]);
        for(size_t i = 0; i < cell_param_count; ++i) {
            if(param_idx < total_params) {
                params[param_idx++] = cell_params[i];
            } else {
                NNL2_ERROR("Parameter index overflow");
                free(cell_params);
                free(params);
                return NULL;
            }
        }
        
        free(cell_params);
    }
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("[RNN] Retrieved %zu parameters from %d layers", total_params, rnn->num_layers);
        NNL2_FUNC_EXIT();
    #endif
    
    return params;
}

/** @brief 
 * Returns the total number of trainable parameter tensors in a Recurrent Neural Network (RNN) layer
 *
 ** @param rnn 
 * Pointer to the RNN layer
 *
 ** @return 
 * Total number of parameter tensors across all RNN cells
 *
 ** @see nnl2_nn_rnn_get_parameters
 **/
size_t nnl2_nn_rnn_get_num_parameters(nnl2_nn_rnn* rnn) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!rnn) {
            NNL2_ERROR("In function nnl2_nn_rnn_get_num_parameters, nnl2_nn_rnn* rnn is NULL");
            return 0;
        }
        
        if(rnn->metadata.nn_magic != NNL2_NN_MAGIC) {
            NNL2_ERROR("Invalid RNN structure (bad magic number)");
            return 0;
        }
    #endif
    
    size_t total_params = 0;
    
    for(int i = 0; i < rnn->num_layers; ++i) {
        if(rnn->cells[i]) {
            total_params += nnl2_nn_rnn_cell_get_num_parameters(rnn->cells[i]);
        } else {
            NNL2_WARN("RNN cell at layer %d is NULL", i);
        }
    }
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_DEBUG("[RNN] Total parameters: %zu (layers: %d, bias: %s)", total_params, rnn->num_layers, rnn->metadata.use_bias ? "yes" : "no");
        NNL2_FUNC_EXIT();
    #endif
    
    return total_params;
}

#endif /** NNL2_NN_RNN_BACKEND_H **/
