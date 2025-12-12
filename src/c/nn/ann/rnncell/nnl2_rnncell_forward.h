#ifndef NNL2_RNNCELL_FORWARD_H
#define NNL2_RNNCELL_FORWARD_H

// NNL2

/** @brief 
 * Performs forward pass of a RNN cell using weights and bias
 * 
 ** @param cell 
 * Pointer to the RNN cell structure containing weights and biases
 *
 ** @param input 
 * Input tensor for the current time step
 *
 ** @param hidden 
 * Hidden state tensor from the previous time step (can be NULL or invalid)
 * 
 ** @return nnl2_ad_tensor*
 * New hidden state tensor after applying RNN cell operations
 *
 ** @note 
 * Caller is responsible for freeing the returned tensor
 */
nnl2_ad_tensor* nnl2_nn_rnn_cell_forward_with_bias(nnl2_nn_rnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden) {
	nnl2_ad_tensor* hidden_safe = hidden;
	bool tmp_hidden_created = false;
	
	if(hidden -> magic_number != TENSOR_MAGIC_ALIVE) {
		hidden_safe = nnl2_ad_zeros_like(input);
		tmp_hidden_created = true;
	}
	
    nnl2_ad_tensor* input_part  = nnl2_ad_gemmvp(input, cell->wxh, cell->bxh, nnl2_ad_reverse_mode, true);
    nnl2_ad_tensor* hidden_part = nnl2_ad_gemmvp(hidden_safe, cell->whh, cell->bhh, nnl2_ad_reverse_mode, true);

    nnl2_quick_print_tensor(hidden->data);

    nnl2_ad_tensor* combined = nnl2_ad_add(input_part, hidden_part, nnl2_ad_reverse_mode, true);

    nnl2_ad_tensor* new_hidden = nnl2_ad_sigmoid(combined, true, nnl2_ad_reverse_mode);

    nnl2_free_ad_tensor(input_part);
    nnl2_free_ad_tensor(hidden_part);

    if(new_hidden != combined)  nnl2_free_ad_tensor(combined);
	if(tmp_hidden_created)  nnl2_free_ad_tensor(hidden_safe);

    return new_hidden;
}

/** @brief 
 * Performs forward pass of a RNN cell using weights only (no bias)
 * 
 ** @param cell 
 * Pointer to the RNN cell structure containing weights
 *
 ** @param input 
 * Input tensor for the current time step
 *
 ** @param hidden 
 * Hidden state tensor from the previous time step (can be NULL or invalid)
 * 
 ** @return nnl2_ad_tensor* 
 * New hidden state tensor after applying RNN cell operations
 *
 ** @note 
 * Caller is responsible for freeing the returned tensor
 */
nnl2_ad_tensor* nnl2_nn_rnn_cell_forward_no_bias(nnl2_nn_rnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden) {
    nnl2_ad_tensor* hidden_safe = hidden;
    bool tmp_hidden_created = false;

    if(!hidden || hidden->magic_number != TENSOR_MAGIC_ALIVE) {
        hidden_safe = nnl2_ad_zeros_like(input);
        tmp_hidden_created = true;
    }

    nnl2_ad_tensor* input_part = nnl2_ad_gemm(input, cell->wxh, nnl2_ad_reverse_mode, true);
    nnl2_ad_tensor* hidden_part = nnl2_ad_gemm(hidden_safe, cell->whh, nnl2_ad_reverse_mode, true);

    nnl2_quick_print_tensor(hidden_safe->data);

    nnl2_ad_tensor* combined = nnl2_ad_add(input_part, hidden_part, nnl2_ad_reverse_mode, true);
    nnl2_ad_tensor* new_hidden = nnl2_ad_sigmoid(combined, true, nnl2_ad_reverse_mode);

    nnl2_free_ad_tensor(input_part);
    nnl2_free_ad_tensor(hidden_part);
	
    if(new_hidden != combined) nnl2_free_ad_tensor(combined);
    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);

    return new_hidden;
}

#endif /** NNL2_RNNCELL_FORWARD_H **/
