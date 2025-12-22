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
	cell->metadata.nn_magic = NNL2_NN_MAGIC;

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
	cell->metadata.nn_magic = NNL2_NN_MAGIC;
    
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

/**
 * @brief 
 * Encodes RNN cell information in nnlrepr format
 * 
 * @param cell 
 * Pointer to RNN cell structure
 * 
 * @return nnl2_nnlrepr_template* 
 * Pointer to created template or NULL on error
 */
static nnl2_nnlrepr_template* nnl2_nn_rnn_cell_nnlrepr_template(nnl2_nn_rnn_cell* cell) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if (!cell || !cell->whh || !cell->whh->data) {
            NNL2_ERROR("In function nnl2_nn_rnn_cell_nnlrepr_template, failed assertion ```!cell || !cell->whh || !cell->whh->data```. returning NULL");
            return NULL;
        }
    #endif
	
    nnl2_nnlrepr_template* result = malloc(sizeof(nnl2_nnlrepr_template));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if (!result) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
    
    // Common metadata
    result->nn_type = nnl2_nn_type_rnn_cell;
    result->num_shapes = 0;
    result->vector_size = 0;
    result->num_childrens = 0;
    result->childrens = NULL;
    result->shapes = NULL;
    result->additional_data = NULL;
    result->dtype = cell->whh->data->dtype;
	
    int num_tensors = 0;
	
    // Count available tensors
    if (cell->wxh && cell->wxh->data)  num_tensors++;
    if (cell->whh && cell->whh->data)  num_tensors++;
    if (cell->bxh && cell->bxh->data)  num_tensors++;
    if (cell->bhh && cell->bhh->data)  num_tensors++;
    
    if(num_tensors == 0) { // if ((bar == 1) == true) { return true; } else { return false; }
        #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
            NNL2_WARN("RNN cell has no tensor data. Returning early");
            NNL2_FUNC_EXIT();
        #endif
		
        return result; 
    }
    
    result->num_shapes = num_tensors;
    result->shapes = malloc(sizeof(int32_t*) * result->num_shapes);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
        if (!result->shapes) {
            NNL2_MALLOC_ERROR();
            free(result);
            return NULL;
        }
		
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
            for (size_t i = 0; i < result->num_shapes; i++) {
                result->shapes[i] = NULL;
            }
        #endif 
    #endif
    
    int shape_index = 0;
    int allocation_failed = 0;
    
    // wxh
    if (cell->wxh && cell->wxh->data && !allocation_failed) {
        int dims = 2; 
        result->shapes[shape_index] = malloc(sizeof(int32_t) * dims);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if (!result->shapes[shape_index]) {
                NNL2_MALLOC_ERROR();
                allocation_failed = 1;
            }
        #endif
        
        if (!allocation_failed) {
            result->shapes[shape_index][0] = cell->wxh->data->shape[0];
            result->shapes[shape_index][1] = cell->wxh->data->shape[1];
            result->vector_size += nnl2_product(cell->wxh->data->shape, dims);
            
            #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
                NNL2_INFO("Added wxh shape: [%d, %d]", 
                          cell->wxh->data->shape[0], 
                          cell->wxh->data->shape[1]);
            #endif
        }
		
        shape_index++;
    }
    
    // whh 
    if(cell->whh && cell->whh->data && !allocation_failed) {
        int dims = 2;
        result->shapes[shape_index] = malloc(sizeof(int32_t) * dims);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if(!result->shapes[shape_index]) {
                NNL2_MALLOC_ERROR();
                allocation_failed = 1;
            }
        #endif
        
        if(!allocation_failed) {
            result->shapes[shape_index][0] = cell->whh->data->shape[0];
            result->shapes[shape_index][1] = cell->whh->data->shape[1];
            result->vector_size += nnl2_product(cell->whh->data->shape, dims);
            
            #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
                NNL2_INFO("Added whh shape: [%d, %d]", 
                          cell->whh->data->shape[0], 
                          cell->whh->data->shape[1]);
            #endif
        }
		
        shape_index++;
    }
    
    // bxh
    if(cell->bxh && cell->bxh->data && !allocation_failed) {
        int dims = 1; 
        result->shapes[shape_index] = malloc(sizeof(int32_t) * dims);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if(!result->shapes[shape_index]) {
                NNL2_MALLOC_ERROR();
                allocation_failed = 1;
            }
        #endif
        
        if(!allocation_failed) {
            result->shapes[shape_index][0] = cell->bxh->data->shape[0];
            result->vector_size += cell->bxh->data->shape[0];
            
            #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
                NNL2_INFO("Added bxh shape: [%d]", cell->bxh->data->shape[0]);
            #endif
        }
		
        shape_index++;
    }

    // bhh 
    if (cell->bhh && cell->bhh->data && !allocation_failed) {
        int dims = 1;
        result->shapes[shape_index] = malloc(sizeof(int32_t) * dims);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            if (!result->shapes[shape_index]) {
                NNL2_MALLOC_ERROR();
                allocation_failed = 1;
            }
        #endif
        
        if (!allocation_failed) {
            result->shapes[shape_index][0] = cell->bhh->data->shape[0];
            result->vector_size += cell->bhh->data->shape[0];
            
            #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
                NNL2_INFO("Added bhh shape: [%d]", cell->bhh->data->shape[0]);
            #endif
        }
		
        shape_index++;
    }
    
    // Cleanup on allocation failure
    if(allocation_failed) {
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
            for (int i = 0; i < shape_index; i++) {
                if (result->shapes[i]) {
                    free(result->shapes[i]);
                }
            }
			
            free(result->shapes);
            free(result);
        #endif
		
        return NULL;
    }
	
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Created RNN cell nnlrepr template with %d tensors, total vector size: %d", num_tensors, result->vector_size);
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief 
 * Decodes RNN cell information from nnlrepr format
 *
 ** @param vector 
 * Encoded 1D nnlrepr vector 
 *
 ** @param offset
 * Encoded vector shift to nnl2_ad_vector_as_parameter(..., ..., offset, ...);
 * 
 ** @param num_shapes
 * Number of shapes of all parameters (2 or 4)
 *
 ** @param shape_wxh 
 * wxh parameter shape [hidden_size, input_size]
 *
 ** @param shape_whh 
 * whh parameter shape [hidden_size, hidden_size]
 *
 ** @param shape_bhh 
 * bhh parameter shape [hidden_size]
 *
 ** @param shape_bxh 
 * bxh parameter shape [hidden_size]
 *
 ** @param dtype 
 * Encoder data type
 *
 ** @return nnl2_nn_rnn_cell* 
 * Pointer to created RNN cell or NULL on error
 */
static nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_nnlrepr_decode(nnl2_ad_tensor* vector, size_t offset, int num_shapes, 
                                                  int32_t* shape_wxh, int32_t* shape_whh, 
                                                  int32_t* shape_bhh, int32_t* shape_bxh, 
                                                  nnl2_tensor_type dtype) {
													  
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(vector, "In function nnl2_nn_rnn_cell_nnlrepr_decode, nnl2_ad_tensor* vector is NULL. returning NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_wxh, "In function nnl2_nn_rnn_cell_nnlrepr_decode, shape_wxh is NULL. returning NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_whh, "In function nnl2_nn_rnn_cell_nnlrepr_decode, shape_whh is NULL. returning NULL", NULL);
        
        if(num_shapes > 2) {
            NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_bhh, "In function nnl2_nn_rnn_cell_nnlrepr_decode, shape_bhh is NULL but bias expected. returning NULL", NULL);
            NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(shape_bxh, "In function nnl2_nn_rnn_cell_nnlrepr_decode, shape_bxh is NULL but bias expected. returning NULL", NULL);
        }
    #endif
    
    // Extract wxh (input-hidden weights)
    nnl2_ad_tensor* wxh_view = nnl2_ad_vector_as_parameter(shape_wxh, 2, offset, vector);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (wxh_view == NULL) {
            NNL2_ERROR("Failed to create wxh view. returning NULL");
            return NULL;
        }
    #endif
    
    offset += nnl2_product(shape_wxh, 2);
    
    // Extract whh (hidden-hidden weights)
    nnl2_ad_tensor* whh_view = nnl2_ad_vector_as_parameter(shape_whh, 2, offset, vector);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(whh_view == NULL) {
            NNL2_ERROR("Failed to create whh view. returning NULL");
            nnl2_free_ad_tensor(wxh_view);
            return NULL;
        }
    #endif
    
    offset += nnl2_product(shape_whh, 2);
    
    // Extract biases if present
    nnl2_ad_tensor* bxh_view = NULL;
    nnl2_ad_tensor* bhh_view = NULL;
    
    if (num_shapes > 2) {
        // Extract bxh (input bias)
        bxh_view = nnl2_ad_vector_as_parameter(shape_bxh, 1, offset, vector);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if (bxh_view == NULL) {
                NNL2_ERROR("Failed to create bxh view. returning NULL");
                nnl2_free_ad_tensor(wxh_view);
                nnl2_free_ad_tensor(whh_view);
                return NULL;
            }
        #endif
        
        offset += shape_bxh[0];
        
        // Extract bhh (hidden bias)
        bhh_view = nnl2_ad_vector_as_parameter(shape_bhh, 1, offset, vector);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if (bhh_view == NULL) {
                NNL2_ERROR("Failed to create bhh view. returning NULL");
                nnl2_free_ad_tensor(wxh_view);
                nnl2_free_ad_tensor(whh_view);
                nnl2_free_ad_tensor(bxh_view);
                return NULL;
            }
        #endif
    }
    
    int input_size = shape_wxh[1];      // wxh shape [hidden_size, input_size]
    int hidden_size = shape_whh[0];     // whh shape [hidden_size, hidden_size]
    bool has_bias = (num_shapes > 2);
    
    nnl2_nn_rnn_cell* result = nnl2_nn_rnn_cell_manual_create(
        input_size, 
        hidden_size, 
        has_bias, 
        dtype, 
        wxh_view, 
        whh_view, 
        bxh_view, 
        bhh_view, 
        nnl2_nn_handle_as_view
    );
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (result == NULL) {
            NNL2_ERROR("Failed to create RNN cell from decoded parameters. returning NULL");
            nnl2_free_ad_tensor(wxh_view);
            nnl2_free_ad_tensor(whh_view);
            if (bxh_view != NULL) nnl2_free_ad_tensor(bxh_view);
            if (bhh_view != NULL) nnl2_free_ad_tensor(bhh_view);
            return NULL;
        }
    #endif
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Successfully decoded RNN cell: input_size=%d, hidden_size=%d, has_bias=%s", input_size, hidden_size, has_bias ? "true" : "false");
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

/** @brief
 * Performs uniform crossover between two RNN Cell layers to create a child layer
 *
 ** @param parent_x
 * Pointer to the first parent RNN Cell layer
 *
 ** @param parent_y
 * Pointer to the second parent RNN Cell layer
 *
 ** @param crossover_rate
 * Probability (0.0 to 1.0) of selecting elements from parent_x
 *
 ** @return nnl2_nn_rnn_cell*
 * Pointer to the newly created child RNN Cell layer
 *
 ** @retval NULL
 * if memory allocation fails, parents are incompatible, or crossover fails
 */
nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_crossover_uniform(nnl2_nn_rnn_cell* parent_x, nnl2_nn_rnn_cell* parent_y, float crossover_rate) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (!parent_x || !parent_y) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parent layers cannot be NULL");
            return NULL;
        }
        
        if (parent_x->metadata.nn_type != nnl2_nn_type_rnn_cell || parent_y->metadata.nn_type != nnl2_nn_type_rnn_cell) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, both parents must be RNN Cell layers");
            return NULL;
        }
        
        if (crossover_rate < 0.0f || crossover_rate > 1.0f) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, crossover_rate must be between 0.0 and 1.0");
            return NULL;
        }
    #endif

    bool use_bias = parent_x->metadata.use_bias;
    if (parent_y->metadata.use_bias != use_bias) {
        NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parents must have same use_bias setting");
        return NULL;
    }

    // Get tensor dimensions from parent_x
    int input_size = parent_x->wxh->data->shape[0];
    int hidden_size = parent_x->wxh->data->shape[1];
    
    // Verify parent_y has same dimensions
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        // wxh dimensions
        if (parent_y->wxh->data->shape[0] != input_size || parent_y->wxh->data->shape[1] != hidden_size) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parents wxh dimensions mismatch");
            return NULL;
        }
        
        // whh dimensions
        if (parent_y->whh->data->shape[0] != hidden_size || parent_y->whh->data->shape[1] != hidden_size) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parents whh dimensions mismatch");
            return NULL;
        }
        
        if (use_bias) {
            // bxh dimensions
            if (parent_y->bxh->data->shape[0] != hidden_size) {
                NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parent bxh dimensions mismatch");
                return NULL;
            }
            
            // bhh dimensions
            if (parent_y->bhh->data->shape[0] != hidden_size) {
                NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, parent bhh dimensions mismatch");
                return NULL;
            }
        }
    #endif

    // Allocate memory for child layer
    nnl2_nn_rnn_cell* cell = malloc(sizeof(nnl2_nn_rnn_cell));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!cell) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif

    // metadata
    cell->metadata.nn_type = nnl2_nn_type_rnn_cell;
    cell->metadata.use_bias = use_bias;
    cell->metadata.nn_magic = NNL2_NN_MAGIC;
    cell->hidden_size = hidden_size;
    cell->forward = parent_x->forward;

    // Crossover for wxh weights
    nnl2_ad_tensor* child_wxh = nnl2_ad_nn_ga_crossover_uniform(parent_x->wxh, parent_y->wxh, crossover_rate);
    if (!child_wxh) {
        NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, wxh crossover failed");
        free(cell);
        return NULL;
    }
    
    cell->wxh = child_wxh;

    // Crossover for whh weights
    nnl2_ad_tensor* child_whh = nnl2_ad_nn_ga_crossover_uniform(parent_x->whh, parent_y->whh, crossover_rate);
    if (!child_whh) {
        NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, whh crossover failed");
        nnl2_free_ad_tensor(child_wxh);
        free(cell);
        return NULL;
    }
    
    cell->whh = child_whh;

    // Crossover for biases if needed
    if (use_bias) {
        // Crossover for bxh bias
        nnl2_ad_tensor* child_bxh = nnl2_ad_nn_ga_crossover_uniform(parent_x->bxh, parent_y->bxh, crossover_rate);
        if (!child_bxh) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, bxh crossover failed");
            nnl2_free_ad_tensor(child_wxh);
            nnl2_free_ad_tensor(child_whh);
            free(cell);
            return NULL;
        }
        
        cell->bxh = child_bxh;

        // Crossover for bhh bias
        nnl2_ad_tensor* child_bhh = nnl2_ad_nn_ga_crossover_uniform(parent_x->bhh, parent_y->bhh, crossover_rate);
        if (!child_bhh) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_crossover_uniform, bhh crossover failed");
            nnl2_free_ad_tensor(child_wxh);
            nnl2_free_ad_tensor(child_whh);
            nnl2_free_ad_tensor(child_bxh);
            free(cell);
            return NULL;
        }
        
        cell->bhh = child_bhh;
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return cell;
}

/** @brief
 * Performs uniform mutation on an RNN Cell layer to create a mutated child layer
 * Creates a new RNN Cell layer where weights (and optionally biases) are mutated
 * by adding random values within [-delta, delta] based on mutation rate
 *
 ** @param parent
 * Pointer to the parent RNN Cell layer
 *
 ** @param mutate_rate
 * Probability (0.0 to 1.0) of mutating each element
 *
 ** @param delta
 * Maximum absolute value of mutation to be added to elements
 *
 ** @return nnl2_nn_rnn_cell*
 * Pointer to the newly created mutated RNN Cell layer
 *
 ** @retval NULL
 * if memory allocation fails or mutation fails
 */
nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_mutation_uniform(nnl2_nn_rnn_cell* parent, float mutate_rate, float delta) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif

    // Safety checks
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MODERATE
        if (!parent) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, parent layer cannot be NULL");
            return NULL;
        }
        
        if (parent->metadata.nn_type != nnl2_nn_type_rnn_cell) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, parent must be an RNN Cell layer");
            return NULL;
        }
        
        if (mutate_rate < 0.0f || mutate_rate > 1.0f) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, mutate_rate must be between 0.0 and 1.0");
            return NULL;
        }
        
        if (delta < 0.0f) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, delta must be non-negative");
            return NULL;
        }
    #endif

    bool use_bias = parent->metadata.use_bias;

    nnl2_nn_rnn_cell* cell = malloc(sizeof(nnl2_nn_rnn_cell));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (!cell) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif

    // metadata
    cell->metadata.nn_type = nnl2_nn_type_rnn_cell;
    cell->metadata.use_bias = use_bias;
    cell->metadata.nn_magic = NNL2_NN_MAGIC;
    cell->hidden_size = parent->hidden_size;
    
    cell->forward = parent->forward;

    // mutation for wxh weights (input-to-hidden)
    nnl2_ad_tensor* mutated_wxh = nnl2_ad_nn_ga_mutation_uniform(parent->wxh, mutate_rate, delta);
    if (!mutated_wxh) {
        NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, wxh weight mutation failed");
        free(cell);
        return NULL;
    }
    
    cell->wxh = mutated_wxh;

    // mutation for whh weights (hidden-to-hidden)
    nnl2_ad_tensor* mutated_whh = nnl2_ad_nn_ga_mutation_uniform(parent->whh, mutate_rate, delta);
    if (!mutated_whh) {
        NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, whh weight mutation failed");
        nnl2_free_ad_tensor(mutated_wxh);  // Clean up
        free(cell);
        return NULL;
    }
    
    cell->whh = mutated_whh;

    // mutation for biases if needed
    if (use_bias) {
        // mutation for bxh bias (input-to-hidden)
        nnl2_ad_tensor* mutated_bxh = nnl2_ad_nn_ga_mutation_uniform(parent->bxh, mutate_rate, delta);
        if (!mutated_bxh) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, bxh bias mutation failed");
            nnl2_free_ad_tensor(mutated_wxh);
            nnl2_free_ad_tensor(mutated_whh);
            free(cell);
            return NULL;
        }
        
        cell->bxh = mutated_bxh;

        // mutation for bhh bias (hidden-to-hidden)
        nnl2_ad_tensor* mutated_bhh = nnl2_ad_nn_ga_mutation_uniform(parent->bhh, mutate_rate, delta);
        if (!mutated_bhh) {
            NNL2_ERROR("In nnl2_nn_rnn_cell_mutation_uniform, bhh bias mutation failed");
            nnl2_free_ad_tensor(mutated_wxh);
            nnl2_free_ad_tensor(mutated_whh);
            nnl2_free_ad_tensor(mutated_bxh);
            free(cell);
            return NULL;
        }
        
        cell->bhh = mutated_bhh;
    } else {
        cell->bxh = NULL;
        cell->bhh = NULL;
    }

    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif

    return cell;
}

/** @brief 
 * Creates a deep copy of a RNN Cell (rnncell) layer
 *
 ** @param src 
 * Pointer to the source RNN Cell layer to be copied
 *
 ** @return
 * A pointer to the newly created deep copy of the RNN Cell layer
 *
 ** @retval NULL 
 * if memory allocation or tensor copying fails
 *
 ** @warning 
 * The caller is responsible for freeing the memory by calling
 * `void nnl2_nn_rnn_cell_free(nnl2_nn_rnn_cell* cell)` on the returned pointer
 *
 ** @see nnl2_nn_rnn_cell_free
 ** @see nnl2_nn_rnn_cell_create
 ** @see nnl2_nn_rnn_cell_manual_create
 **/
nnl2_nn_rnn_cell* nnl2_nn_rnn_cell_deep_copy(const nnl2_nn_rnn_cell* src) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(src, "In function nnl2_nn_rnn_cell_deep_copy, const nnl2_nn_rnn_cell* src is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(src->wxh, "In function nnl2_nn_rnn_cell_deep_copy, src->wxh is NULL", NULL);
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(src->whh, "In function nnl2_nn_rnn_cell_deep_copy, src->whh is NULL", NULL);
    #endif
    
    nnl2_nn_rnn_cell* dst = malloc(sizeof(nnl2_nn_rnn_cell));
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!dst) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
    
    // Copy metadata and basic fields
    dst->metadata = src->metadata;
    dst->hidden_size = src->hidden_size;
    dst->forward = src->forward;
    
    // Deep copy wxh tensor (input-to-hidden weights)
    dst->wxh = nnl2_ad_copy(src->wxh, src->wxh->data->dtype);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!dst->wxh) {
            NNL2_ERROR("Failed to copy wxh tensor in nnl2_nn_rnn_cell_deep_copy");
            free(dst);
            return NULL;
        }
    #endif
    
    // Deep copy whh tensor (hidden-to-hidden weights)
    dst->whh = nnl2_ad_copy(src->whh, src->whh->data->dtype);
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!dst->whh) {
            NNL2_ERROR("Failed to copy whh tensor in nnl2_nn_rnn_cell_deep_copy");
            nnl2_free_ad_tensor(dst->wxh);
            free(dst);
            return NULL;
        }
    #endif
    
    // Deep copy bias tensors if they exist
    if(src->metadata.use_bias && src->bxh && src->bhh) {
        dst->bxh = nnl2_ad_copy(src->bxh, src->bxh->data->dtype);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!dst->bxh) {
                NNL2_ERROR("Failed to copy bxh tensor in nnl2_nn_rnn_cell_deep_copy");
                nnl2_free_ad_tensor(dst->wxh);
                nnl2_free_ad_tensor(dst->whh);
                free(dst);
                return NULL;
            }
        #endif
        
        dst->bhh = nnl2_ad_copy(src->bhh, src->bhh->data->dtype);
        
        #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
            if(!dst->bhh) {
                NNL2_ERROR("Failed to copy bhh tensor in nnl2_nn_rnn_cell_deep_copy");
                nnl2_free_ad_tensor(dst->wxh);
                nnl2_free_ad_tensor(dst->whh);
                nnl2_free_ad_tensor(dst->bxh);
                free(dst);
                return NULL;
            }
        #endif
    } else {
        dst->bxh = NULL;
        dst->bhh = NULL;
    }
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Successfully created deep copy of RNN Cell layer");
        NNL2_DEBUG("Copied wxh shape: [%d, %d], whh shape: [%d, %d]", 
                   dst->wxh->data->shape[0], dst->wxh->data->shape[1],
                   dst->whh->data->shape[0], dst->whh->data->shape[1]);
        if(dst->metadata.use_bias) {
            NNL2_DEBUG("Copied bxh shape: [%d], bhh shape: [%d]", 
                       dst->bxh->data->shape[0],
                       dst->bhh->data->shape[0]);
        } else {
            NNL2_DEBUG("No bias tensors");
        }
    #endif
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return dst;
}
	
#endif /** NNL2_RNNCELL_BACKEND_H **/
