#ifndef NNL2_UNIRNNCELL_BACKEND_H
#define NNL2_UNIRNNCELL_BACKEND_H

// NNL2

/** @file nnl2_unirnncell_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Uni-directional RNN Cell (unirnncell) Backend
 **
 ** Contains the entire basic backend for unirnncell, including
 ** the structure, allocators, forward function pointers, and
 ** other auxiliary functions
 **/

///@{ [nnl2_nn_unirnn_cell]

/** @brief Uni-directional RNN Cell layer structure
 ** @see nnl2_nn_ann
 ** @extends nnl2_nn_ann
 **/
typedef struct nnl2_nn_unirnn_cell_struct {
    nnl2_nn_ann metadata;     ///< Base neural network metadata and type info
    nnl2_ad_tensor* wxh;      ///< Input-to-hidden weights
    nnl2_ad_tensor* whh;      ///< Hidden-to-hidden weights
    nnl2_ad_tensor* bxh;      ///< Input bias
    nnl2_ad_tensor* bhh;      ///< Hidden bias
    int hidden_size;		  ///< Hidden state size
	
	nnl2_ad_tensor* (*forward)(		///< Forward propagation function pointer
	    struct nnl2_nn_unirnn_cell_struct* cell,
        nnl2_ad_tensor* input,
        nnl2_ad_tensor* hidden
	);
} nnl2_nn_unirnn_cell;

///@} [nnl2_nn_unirnn_cell]



/** @brief Initializes unirnncell with zeros **/
static bool unirnncell_init_zeros(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with zeros **/
static bool unirnncell_init_zeros(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with uniform random values **/
static bool unirnncell_init_rand(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with normal random values **/
static bool unirnncell_init_randn(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with Xavier normal initialization **/
static bool unirnncell_init_xavier_normal(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with Xavier uniform initialization **/
static bool unirnncell_init_xavier_uniform(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with Kaiming normal initialization **/
static bool unirnncell_init_kaiming_normal(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with Kaiming uniform initialization **/
static bool unirnncell_init_kaiming_uniform(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Initializes unirnncell with identity matrix for testing **/
static bool unirnncell_init_identity(nnl2_nn_unirnn_cell* cell, int input_size, int hidden_size, nnl2_tensor_type dtype);

/** @brief Forward pass using weights and biases **/
nnl2_ad_tensor* nnl2_nn_unirnn_cell_forward_with_bias(nnl2_nn_unirnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden);

/** @brief Forward pass using weights only (no bias) **/
nnl2_ad_tensor* nnl2_nn_unirnn_cell_forward_no_bias(nnl2_nn_unirnn_cell* cell, nnl2_ad_tensor* input, nnl2_ad_tensor* hidden);



/** @brief 
 * Creates a unidirectional RNN cell (unirnncell)
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
 * Pointer to the newly created unirnncell structure
 *
 ** @retval
 * NULL if memory allocation or initialization fails
 **
 ** @see nnl2_nn_unirnn_cell_free
 **/
nnl2_nn_unirnn_cell* nnl2_nn_unirnn_cell_create(int input_size, int hidden_size, bool bias, nnl2_tensor_type dtype, nnl2_nn_init_type init_type) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_nn_unirnn_cell* cell = malloc(sizeof(nnl2_nn_unirnn_cell));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if(!cell) {
            NNL2_MALLOC_ERROR();
            return NULL;
        }
    #endif
	
	cell->metadata.nn_type = nnl2_nn_type_unirnn_cell;
    cell->metadata.use_bias = bias;
	
	cell->hidden_size = hidden_size;
    
    bool init_success = false;
    
	switch(init_type) {
        case nnl2_nn_init_zeros:             init_success = unirnncell_init_zeros(cell, input_size, hidden_size, dtype);            break;
        case nnl2_nn_init_rand:              init_success = unirnncell_init_rand(cell, input_size, hidden_size, dtype);             break;
        case nnl2_nn_init_randn:             init_success = unirnncell_init_randn(cell, input_size, hidden_size, dtype);            break;
        case nnl2_nn_init_xavier_normal:     init_success = unirnncell_init_xavier_normal(cell, input_size, hidden_size, dtype);    break;
        case nnl2_nn_init_xavier_uniform:    init_success = unirnncell_init_xavier_uniform(cell, input_size, hidden_size, dtype);   break;
        case nnl2_nn_init_kaiming_normal:    init_success = unirnncell_init_kaiming_normal(cell, input_size, hidden_size, dtype);   break;
        case nnl2_nn_init_kaiming_uniform:   init_success = unirnncell_init_kaiming_uniform(cell, input_size, hidden_size, dtype);  break;
        case nnl2_nn_init_identity:          init_success = unirnncell_init_identity(cell, input_size, hidden_size, dtype);         break;
        
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
            NNL2_ERROR("UniRNN cell initialization failed for type: %d", init_type);
        #endif
		
        free(cell);
        return NULL;
    }
	
	cell->forward = cell->metadata.use_bias ? nnl2_nn_unirnn_cell_forward_with_bias : nnl2_nn_unirnn_cell_forward_no_bias;
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
	return cell;
}

/** @brief 
 * Frees memory used by a unirnncell layer
 *
 ** @param cell
 * Pointer to the unirnncell to free
 */
void nnl2_nn_unirnn_cell_free(nnl2_nn_unirnn_cell* cell) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN(cell, "In function nnl2_nn_unirnn_cell_free, nnl2_nn_unirnn_cell* cell is NULL");
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
 * Retrieves all trainable parameters from a unirnncell
 *
 ** @param cell
 * Pointer to the unirnncell
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
nnl2_ad_tensor** nnl2_nn_unirnn_cell_get_parameters(nnl2_nn_unirnn_cell* cell) { 
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_unirnn_cell_get_parameters, nnl2_nn_unirnn_cell* cell is NULL", NULL);
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
 * Returns the number of trainable parameter tensors in a unirnncell
 *
 ** @param cell
 * Pointer to the unirnncell
 *
 ** @return
 * Number of parameter tensors (2 if no bias, 4 if bias is used)
 */
size_t nnl2_nn_unirnn_cell_get_num_parameters(nnl2_nn_unirnn_cell* cell) {
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_unirnn_cell_get_num_parameters, nnl2_nn_unirnn_cell* cell is NULL", 0);
    #endif 
	
    size_t num_parameters = cell->metadata.use_bias ? 4 : 2;
	
	#if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
	
    return num_parameters;
}

/** @brief 
 * Get input size of a unirnncell (number of input features)
 *
 ** @param cell
 * Pointer to the unirnncell structure
 *
 ** @return int
 * Number of input features (first dimension of wxh)
 **/
int nnl2_nn_unirnn_cell_get_input_size(nnl2_nn_unirnn_cell* cell) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_unirnn_cell_get_input_size, nnl2_nn_unirnn_cell* cell is NULL", 0);
    #endif 
    
    int input_size = cell->wxh ? cell->wxh->data->shape[0] : 0;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return input_size;
}


/** @brief 
 * Get hidden size of a unirnncell (number of hidden units)
 *
 ** @param cell
 * Pointer to the unirnncell structure
 *
 ** @return int
 * Number of hidden units (second dimension of whh)
 */
int nnl2_nn_unirnn_cell_get_hidden_size(nnl2_nn_unirnn_cell* cell) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        NNL2_CHECK_NULL_IF_ERR_RETURN_VAL(cell, "In function nnl2_nn_unirnn_cell_get_hidden_size, nnl2_nn_unirnn_cell* cell is NULL", 0);
    #endif 
    
    int hidden_size = cell->whh ? cell->whh->data->shape[1] : cell->hidden_size;
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return hidden_size;
}

/** @brief 
 * Prints unirnncell information in a human-readable format
 *
 ** @param nn
 * Pointer to the unirnncell to print
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
void nnl2_nn_unirnn_cell_print(nnl2_nn_unirnn_cell* nn, bool terpri) {
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    if(!nn) {
        printf("(rnncell NULL)%s", terpri ? "\n" : "");
        return;
    }

    printf("(rnncell %d -> %d :bias %s :bidirectional nil)%s", 
           nnl2_nn_unirnn_cell_get_input_size(nn), nnl2_nn_unirnn_cell_get_hidden_size(nn), 
           nn->metadata.use_bias ? "t" : "nil", terpri ? "\n" : "");
    
    #if NNL2_DEBUG_MODE > NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
}
	
#endif /** NNL2_UNIRNNCELL_BACKEND_H **/
