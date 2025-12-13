#ifndef NNL2_RNNCELL_BACKEND_H
#define NNL2_RNNCELL_BACKEND_H

// NNL2

/** @file nnl2_rnncell_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief RNN Cell (rnncell) Backend
 **
 ** Contains the entire basic backend for rnncell, including
 ** the structure, allocators, forward function pointers, and
 ** other auxiliary functions
 **
 ** UPD: Renamed from unirnncell to rnncell
 **/

///@{ [nnl2_nn_rnn_cell]

/** @brief RNN Cell layer structure
 ** @see nnl2_nn_ann
 ** @extends nnl2_nn_ann
 **/
typedef struct nnl2_nn_rnn_cell_struct {
    nnl2_nn_ann metadata;     ///< Base neural network metadata and type info
    nnl2_ad_tensor* wxh;      ///< Input-to-hidden weights
    nnl2_ad_tensor* whh;      ///< Hidden-to-hidden weights
    nnl2_ad_tensor* bxh;      ///< Input bias
    nnl2_ad_tensor* bhh;      ///< Hidden bias
    int hidden_size;		  ///< Hidden state size
	
	nnl2_ad_tensor* (*forward)(		///< Forward propagation function pointer
	    struct nnl2_nn_rnn_cell_struct* cell,
        nnl2_ad_tensor* input,
        nnl2_ad_tensor* hidden
	);
} nnl2_nn_rnn_cell;

///@} [nnl2_nn_rnn_cell]



/** @brief Initializes rnncell with zeros **/
static bool rnncell_init_zeros(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with zeros **/
static bool rnncell_init_zeros(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with uniform random values **/
static bool rnncell_init_rand(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with normal random values **/
static bool rnncell_init_randn(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with Xavier normal initialization **/
static bool rnncell_init_xavier_normal(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with Xavier uniform initialization **/
static bool rnncell_init_xavier_uniform(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with Kaiming normal initialization **/
static bool rnncell_init_kaiming_normal(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with Kaiming uniform initialization **/
static bool rnncell_init_kaiming_uniform(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes rnncell with identity matrix for testing **/
static bool rnncell_init_identity(nnl2_nn_rnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Forward pass using weights and biases **/
nnl2_ad_tensor* nnl2_nn_rnn_cell_forward_with_bias(nnl2_nn_rnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden);

/** @brief Forward pass using weights only (no bias) **/
nnl2_ad_tensor* nnl2_nn_rnn_cell_forward_no_bias(nnl2_nn_rnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden);



/** @brief 
 * Creates a RNN cell (rnncell)
 *
 ** @param input_size
 * Number of input features
 *
 ** @param hidden_size
 * Number of hidden units in the cell
 *
 ** @param bias
 * If true, bias tensors (bxh and bhh) are created and initialized
 *
 ** @param dtype
 * Data type for the cell's tensors
 *
 ** @param init_type
 * Initialization type for weights and biases
 *
 ** @return
 * Pointer to the newly created rnncell structure
 *
 ** @retval
 * NULL if memory allocation or initialization fails
 **
 ** @see nnl2_nn_rnn_cell_free
 **/
nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_create(int input_size, int hidden_size, bool bias, nnl2_tensor_type dtype, nnl2_nn_init_type init_type) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_nn_rnn_cell* cell = malloc(sizeof(nnl2_nn_rnn_cell));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
	
	cell->metadata.nn_type = nnl2_nn_type_rnn_cell;
    cell->metadata.use_bias = bias;
	
	cell->hidden_size = hidden_size;
    
    bool init_success = false;
    
	switch(init_type) {
        case nnl2_nn_init_zeros:             init_success = rnncell_init_zeros(cell, input_size, hidden_size, dtype);            break;
        case nnl2_nn_init_rand:              init_success = rnncell_init_rand(cell, input_size, hidden_size, dtype);             break;
        case nnl2_nn_init_randn:             init_success = rnncell_init_randn(cell, input_size, hidden_size, dtype);            break;
        case nnl2_nn_init_xavier_normal:     init_success = rnncell_init_xavier_normal(cell, input_size, hidden_size, dtype);    break;
        case nnl2_nn_init_xavier_uniform:    init_success = rnncell_init_xavier_uniform(cell, input_size, hidden_size, dtype);   break;
        case nnl2_nn_init_kaiming_normal:    init_success = rnncell_init_kaiming_normal(cell, input_size, hidden_size, dtype);   break;
        case nnl2_nn_init_kaiming_uniform:   init_success = rnncell_init_kaiming_uniform(cell, input_size, hidden_size, dtype);  break;
        case nnl2_nn_init_identity:          init_success = rnncell_init_identity(cell, input_size, hidden_size, dtype);         break;
        
        case nnl2_nn_init_unknown:
		
        default: {
			#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                NNL2_ERROR("Unknown initialization type: %d", init_type);
            #endif
			
            free(cell);
            return NULL;
		}
    }
	
	if(!init_success) {
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            NNL2_ERROR("RNN cell initialization failed for type: %d", init_type);
        #endif
		
        free(cell);
        return NULL;
    }
	
	cell->forward = cell->metadata.use_bias ? nnl2_nn_rnn_cell_forward_with_bias : nnl2_nn_rnn_cell_forward_no_bias;
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return cell;
}

/** @brief 
 * Creates a RNN cell (rnncell) with user-provided tensors
 *
 ** @param input_size
 * Number of input features
 *
 ** @param hidden_size
 * Number of hidden units in the cell
 *
 ** @param bias
 * If true, bias tensors (bxh and bhh) are used and must be provided
 *
 ** @param dtype
 * Data type for the cell's tensors
 *
 ** @param wxh
 * Pointer to a user-provided weights tensor for input-to-hidden connections.
 * Must have shape [input_size, hidden_size]
 *
 ** @param whh
 * Pointer to a user-provided weights tensor for hidden-to-hidden connections.
 * Must have shape [hidden_size, hidden_size]
 *
 ** @param bxh
 * Pointer to a user-provided bias tensor for input-to-hidden connections.
 * Must have shape [hidden_size] if bias is true
 *
 ** @param bhh
 * Pointer to a user-provided bias tensor for hidden-to-hidden connections.
 * Must have shape [hidden_size] if bias is true
 *
 ** @param handle_as
 * Enum specifying how the cell handles the provided tensors
 * 
 * Details: 
 *     nnl2_nn_handle_as_copy: make a copy of the provided tensors (safe)
 *     nnl2_nn_handle_as_view: use the provided tensors directly (lifetime managed by the caller)
 *
 ** @return nnl2_nn_rnn_cell*
 * A pointer to the newly created RNN cell
 *
 ** @retval NULL 
 * if memory allocation fails or if tensor shapes are incorrect
 *
 ** @see nnl2_nn_rnn_cell_create
 ** @see nnl2_nn_handle_as
 ** @see nnl2_nn_rnn_cell_free
 **/
nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_manual_create(int input_size, int hidden_size, bool bias,
                                                nnl2_tensor_type dtype,
                                                nnl2_ad_tensor* wxh, nnl2_ad_tensor* whh,
                                                nnl2_ad_tensor* bxh, nnl2_ad_tensor* bhh,
                                                nnl2_nn_handle_as handle_as) {
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    nnl2_nn_rnn_cell* cell = malloc(sizeof(nnl2_nn_rnn_cell));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
    
    // wxh must be [input_size, hidden_size]
    if(wxh->data->shape[0] != input_size || wxh->data->shape[1] != hidden_size) {
        NNL2_ERROR("In function nnl2_nn_rnn_cell_manual_create, wxh tensor shape is NOT CORRECT. "
                   "Expected shape: [%d, %d] (input_size, hidden_size), Got: [%d, %d]",
                   input_size, hidden_size, wxh->data->shape[0], wxh->data->shape[1]);
				   
        free(cell);
        return NULL;
    }
    
    //  whh must be [hidden_size, hidden_size]
    if(whh->data->shape[0] != hidden_size || whh->data->shape[1] != hidden_size) {
        NNL2_ERROR("In function nnl2_nn_rnn_cell_manual_create, whh tensor shape is NOT CORRECT. "
                   "Expected shape: [%d, %d] (hidden_size, hidden_size), Got: [%d, %d]",
                   hidden_size, hidden_size, whh->data->shape[0], whh->data->shape[1]);
				   
        free(cell);
        return NULL;
    }

    if(bias) {
        if(!bxh || !bhh) {
            NNL2_ERROR("In function nnl2_nn_rnn_cell_manual_create, bias is enabled but bxh or bhh tensor is NULL");
            free(cell);
            return NULL;
        }
        
        // bxh must be [hidden_size]
        if(bxh->data->shape[0] != hidden_size) {
            NNL2_ERROR("In function nnl2_nn_rnn_cell_manual_create, bxh tensor shape is NOT CORRECT. "
                       "Expected shape: [%d] (hidden_size), Got: [%d]",
                       hidden_size, bxh->data->shape[0]);
					   
            free(cell);
            return NULL;
        }
        
        // bhh must be [hidden_size]
        if(bhh->data->shape[0] != hidden_size) {
            NNL2_ERROR("In function nnl2_nn_rnn_cell_manual_create, bhh tensor shape is NOT CORRECT. "
                       "Expected shape: [%d] (hidden_size), Got: [%d]",
                       hidden_size, bhh->data->shape[0]);
					   
            free(cell);
            return NULL;
        }
    }
    
    // metadata
    cell->metadata.nn_type = nnl2_nn_type_rnn_cell;
    cell->metadata.use_bias = bias;
    cell->hidden_size = hidden_size;
    
    switch(handle_as) {
        case nnl2_nn_handle_as_copy: {
            cell->wxh = nnl2_ad_copy(wxh, dtype);
            cell->whh = nnl2_ad_copy(whh, dtype);
            
            if(bias) {
                cell->bxh = nnl2_ad_copy(bxh, dtype);
                cell->bhh = nnl2_ad_copy(bhh, dtype);
            }
			
            break;
        }
        
        case nnl2_nn_handle_as_view: {
            cell->wxh = wxh;
            cell->whh = whh;
            
            if(bias) {
                cell->bxh = bxh;
                cell->bhh = bhh;
            }
			
            break;
        }
        
        default: {
            #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
                NNL2_ERROR("Unknown handle method in function nnl2_nn_rnn_cell_manual_create. "
                           "Handle enum type numbering: %d", handle_as);
            #endif
            
            free(cell);
            return NULL;
        }
    }
    
    cell->forward = bias ? nnl2_nn_rnn_cell_forward_with_bias : nnl2_nn_rnn_cell_forward_no_bias;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return cell;
}

/** @brief 
 * Frees memory used by a rnncell layer
 *
 ** @param cell
 * Pointer to the rnncell to free
 */
void nnl2_nn_rnn_cell_free(nnl2_nn_rnn_cell* cell) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(cell, "In function nnl2_nn_rnn_cell_free, nnl2_nn_rnn_cell* cell is NULL");
    #endif 
	
    if(cell->wxh)  nnl2_free_ad_tensor(cell->wxh);
    if(cell->whh)  nnl2_free_ad_tensor(cell->whh);
    if(cell->bxh)  nnl2_free_ad_tensor(cell->bxh);
    if(cell->bhh)  nnl2_free_ad_tensor(cell->bhh);
    
    free(cell);
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}

/** @brief 
 * Retrieves all trainable parameters from a rnncell
 *
 ** @param cell
 * Pointer to the rnncell
 *
 ** @return
 * Dynamically allocated array of parameter tensor pointers
 *
 ** @retval
 * NULL if memory allocation fails
 *
 ** @note
 * Caller is responsible for freeing the returned array
 */
nnl2_ad_tensor** nnl2_nn_rnn_cell_get_parameters(nnl2_nn_rnn_cell* cell) { 
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_rnn_cell_get_parameters, nnl2_nn_rnn_cell* cell is NULL", NULL);
    #endif 
	
    nnl2_ad_tensor** params = malloc(sizeof(nnl2_ad_tensor*) * (cell->metadata.use_bias ? 4 : 2));
    
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!params) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
	
    params[0] = cell->wxh;
    params[1] = cell->whh;
    
    if(cell->metadata.use_bias) {
        params[2] = cell->bxh;
        params[3] = cell->bhh;
    }
    
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
    return params;
}

/** @brief 
 * Returns the number of trainable parameter tensors in a rnncell
 *
 ** @param cell
 * Pointer to the rnncell
 *
 ** @return
 * Number of parameter tensors (2 if no bias, 4 if bias is used)
 */
size_t nnl2_nn_rnn_cell_get_num_parameters(nnl2_nn_rnn_cell* cell) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_rnn_cell_get_num_parameters, nnl2_nn_rnn_cell* cell is NULL", 0);
    #endif 
	
    size_t num_parameters = cell->metadata.use_bias ? 4 : 2;
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
    return num_parameters;
}

/** @brief 
 * Get input size of a rnncell (number of input features)
 *
 ** @param cell
 * Pointer to the rnncell structure
 *
 ** @return int
 * Number of input features (first dimension of wxh)
 **/
int nnl2_nn_rnn_cell_get_input_size(nnl2_nn_rnn_cell* cell) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_rnn_cell_get_input_size, nnl2_nn_rnn_cell* cell is NULL", 0);
    #endif 
    
    int input_size = cell->wxh ? cell->wxh->data->shape[0] : 0;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return input_size;
}


/** @brief 
 * Get hidden size of a rnncell (number of hidden units)
 *
 ** @param cell
 * Pointer to the rnncell structure
 *
 ** @return int
 * Number of hidden units (second dimension of whh)
 */
int nnl2_nn_rnn_cell_get_hidden_size(nnl2_nn_rnn_cell* cell) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_rnn_cell_get_hidden_size, nnl2_nn_rnn_cell* cell is NULL", 0);
    #endif 
    
    int hidden_size = cell->whh ? cell->whh->data->shape[1] : cell->hidden_size;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return hidden_size;
}

/** @brief 
 * Prints rnncell information in a human-readable format
 *
 ** @param nn
 * Pointer to the rnncell to print
 *
 ** @param terpri
 * If true, prints a newline after the output
 *
 ** @note
 * Shows input size (wxh), hidden size (whh), and whether bias is used
 *
 ** @note
 * Part of nnl2.hli.nn:print-model in Lisp interface
 */
void nnl2_nn_rnn_cell_print(nnl2_nn_rnn_cell* nn, bool terpri) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!nn) {
        printf("(rnncell NULL)%s", terpri ? "\n" : "");
        return;
    }

    printf("(rnncell %d -> %d :bias %s)%s", 
           nnl2_nn_rnn_cell_get_input_size(nn), nnl2_nn_rnn_cell_get_hidden_size(nn), 
           nn->metadata.use_bias ? "t" : "nil", terpri ? "\n" : "");
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}
	
#endif /** NNL2_RNNCELL_BACKEND_H **/
