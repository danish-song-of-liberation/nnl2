#ifndef NNL2_NN_RNN_BACKEND_H
#define NNL2_NN_RNN_BACKEND_H

typedef struct nnl2_nn_rnn_struct {
    nnl2_nn_ann metadata;
    nnl2_nn_rnn_cell** cells;
    int hidden_size;
	int num_layers;
} nnl2_nn_rnn;

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
	
	return rnn;
}

#endif /** NNL2_NN_RNN_BACKEND_H **/
