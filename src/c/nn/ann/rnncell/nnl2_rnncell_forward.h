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
    //NNL2_DEBUG("[RNN_CELL] === START ===");
   // NNL2_DEBUG("[RNN_CELL] cell=%p, input=%p, hidden=%p", 
     //          (void*)cell, (void*)input, (void*)hidden);
    //
    //if(!cell) {
     //   NNL2_ERROR("[RNN_CELL] cell is NULL!");
    //    return NULL;
    //}
    //
    //if(!input) {
    //    NNL2_ERROR("[RNN_CELL] input is NULL!");
    //    return NULL;
    //}
    //
    //if(!input->data) {
    //    NNL2_ERROR("[RNN_CELL] input->data is NULL!");
    //    return NULL;
    //}
    
    //if(input->magic_number != TENSOR_MAGIC_ALIVE) {
    //    NNL2_ERROR("[RNN_CELL] input has invalid magic number: %u (expected: %u)", input->magic_number, TENSOR_MAGIC_ALIVE);
    //    return NULL;
    //}
    
    //NNL2_DEBUG("[RNN_CELL] Input tensor:");
    //NNL2_DEBUG("[RNN_CELL]   shape: [%d x %d]", 
    //           input->data->shape[0], input->data->shape[1]);
    //NNL2_DEBUG("[RNN_CELL]   strides: [%zu x %zu]", 
    //           input->data->strides[0], input->data->strides[1]);
    //NNL2_DEBUG("[RNN_CELL]   data pointer: %p", (void*)input->data->data);
    //NNL2_DEBUG("[RNN_CELL]   is_view: %d", input->data->is_view);
    
    nnl2_ad_tensor* hidden_safe = hidden;
    bool tmp_hidden_created = false;
    
  // if(!hidden || hidden->magic_number != TENSOR_MAGIC_ALIVE) {
    //    NNL2_DEBUG("[RNN_CELL] Hidden state is invalid or NULL, creating zeros...");
    //    hidden_safe = nnl2_ad_zeros_like(input);
    //   
    //    if(!hidden_safe) {
    //        NNL2_ERROR("[RNN_CELL] Failed to create zeros tensor!");
     //       return NULL;
    //    }
     //   
     //   tmp_hidden_created = true;
    //    NNL2_DEBUG("[RNN_CELL] Created temporary hidden state: %p", (void*)hidden_safe);
    //}
    
    //if(hidden_safe && hidden_safe->data) {
     //   NNL2_DEBUG("[RNN_CELL] Hidden tensor:");
     //   NNL2_DEBUG("[RNN_CELL]   shape: [%d x %d]", 
      //             hidden_safe->data->shape[0], hidden_safe->data->shape[1]);
    //    NNL2_DEBUG("[RNN_CELL]   strides: [%zu x %zu]", 
    //               hidden_safe->data->strides[0], hidden_safe->data->strides[1]);
    //    NNL2_DEBUG("[RNN_CELL]   data pointer: %p", (void*)hidden_safe->data->data);
    //    NNL2_DEBUG("[RNN_CELL]   is_view: %d", hidden_safe->data->is_view);
   // } else {
    //    NNL2_ERROR("[RNN_CELL] hidden_safe or its data is NULL after creation!");
     //   return NULL;
   // }
    
    //if(!cell->wxh || cell->wxh->magic_number != TENSOR_MAGIC_ALIVE) {
    //    NNL2_ERROR("[RNN_CELL] cell->wxh is invalid!");
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
    
    //if(!cell->whh || cell->whh->magic_number != TENSOR_MAGIC_ALIVE) {
    //    NNL2_ERROR("[RNN_CELL] cell->whh is invalid!");
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
    
    //NNL2_DEBUG("[RNN_CELL] Weight tensors:");
    //NNL2_DEBUG("[RNN_CELL]   wxh shape: [%d x %d]", cell->wxh->data->shape[0], cell->wxh->data->shape[1]);
    //NNL2_DEBUG("[RNN_CELL]   whh shape: [%d x %d]", cell->whh->data->shape[0], cell->whh->data->shape[1]);
    
    // input [batch, input_size] 
    // wxh [input_size, hidden_size] 
    //if(input->data->shape[1] != cell->wxh->data->shape[0]) {
    //    NNL2_ERROR("[RNN_CELL] Dimension mismatch: input cols=%d, wxh rows=%d",
    //               input->data->shape[1], cell->wxh->data->shape[0]);
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
    
    // hidden [batch, hidden_size] 
    // whh [hidden_size, hidden_size] 
    //if(hidden_safe->data->shape[1] != cell->whh->data->shape[0]) {
    //    NNL2_ERROR("[RNN_CELL] Dimension mismatch: hidden cols=%d, whh rows=%d",
    //               hidden_safe->data->shape[1], cell->whh->data->shape[0]);
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
   // }
    
    //NNL2_DEBUG("[RNN_CELL] Calling gemmvp for input part...");
	
    nnl2_ad_tensor* input_part = nnl2_ad_gemmvp(input, cell->wxh, cell->bxh, nnl2_ad_reverse_mode, true);
    //if(!input_part) {
    //    NNL2_ERROR("[RNN_CELL] gemmvp for input failed!");
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
	
    //NNL2_DEBUG("[RNN_CELL] Input part created: %p", (void*)input_part);
    
    //NNL2_DEBUG("[RNN_CELL] Calling gemmvp for hidden part...");
	
    nnl2_ad_tensor* hidden_part = nnl2_ad_gemmvp(hidden_safe, cell->whh, cell->bhh, nnl2_ad_reverse_mode, true);
    //if(!hidden_part) {
    //    NNL2_ERROR("[RNN_CELL] gemmvp for hidden failed!");
    //    nnl2_free_ad_tensor(input_part);
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
	
    //NNL2_DEBUG("[RNN_CELL] Hidden part created: %p", (void*)hidden_part);
    
    //NNL2_DEBUG("[RNN_CELL] Adding parts...");
	
    nnl2_ad_tensor* combined = nnl2_ad_add(input_part, hidden_part, nnl2_ad_reverse_mode, true);
    //if(!combined) {
    //    NNL2_ERROR("[RNN_CELL] add failed!");
    //    nnl2_free_ad_tensor(input_part);
     //   nnl2_free_ad_tensor(hidden_part);
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
    //}
	
    //NNL2_DEBUG("[RNN_CELL] Combined tensor: %p", (void*)combined);
    
    //NNL2_DEBUG("[RNN_CELL] Applying sigmoid...");
	
    nnl2_ad_tensor* new_hidden = nnl2_ad_sigmoid(combined, true, nnl2_ad_reverse_mode);
    //if(!new_hidden) {
       //NNL2_ERROR("[RNN_CELL] sigmoid failed!");
     //   nnl2_free_ad_tensor(combined);
    //    nnl2_free_ad_tensor(input_part);
    //    nnl2_free_ad_tensor(hidden_part);
    //    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);
    //    return NULL;
   // }
	
    //NNL2_DEBUG("[RNN_CELL] New hidden state: %p", (void*)new_hidden);
    
    nnl2_free_ad_tensor(input_part);
    nnl2_free_ad_tensor(hidden_part);
    
    if(new_hidden != combined) {
        nnl2_free_ad_tensor(combined);
    }
    
    if(tmp_hidden_created) {
        nnl2_free_ad_tensor(hidden_safe);
    }
    
    //NNL2_DEBUG("[RNN_CELL] === SUCCESS ===");
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

    nnl2_ad_tensor* combined = nnl2_ad_add(input_part, hidden_part, nnl2_ad_reverse_mode, true);
    nnl2_ad_tensor* new_hidden = nnl2_ad_sigmoid(combined, true, nnl2_ad_reverse_mode);

    nnl2_free_ad_tensor(input_part);
    nnl2_free_ad_tensor(hidden_part);
	
    if(new_hidden != combined) nnl2_free_ad_tensor(combined);
    if(tmp_hidden_created) nnl2_free_ad_tensor(hidden_safe);

    return new_hidden;
}

#endif /** NNL2_RNNCELL_FORWARD_H **/
