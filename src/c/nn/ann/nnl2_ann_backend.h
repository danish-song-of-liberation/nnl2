#ifndef NNL2_ANN_BACKEND_H
#define NNL2_ANN_BACKEND_H 

// NNL2

/** @file nnl2_ann_backend.h
 ** @date 2025
 ** @copyright MIT License
 ** @brief Contains ANN Backend types and utilities
 **
 ** Filepath: src/c/nn/ann/nnl2_ann_backend.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/



#ifndef NNL2_NN_TYPE_DEFINED
#define NNL2_NN_TYPE_DEFINED

///@{ [nnl2_nn_type]

typedef enum {
    nnl2_nn_type_fnn,           ///< Fully Connected Neural Network 
	nnl2_nn_type_rnn_cell,      ///< Vanilla Recurrent Neural Network Cell
	nnl2_nn_type_sequential,    ///< Sequential neural network (layers in sequence)
	nnl2_nn_type_sigmoid,		///< Sigmoid layer
	nnl2_nn_type_tanh,			///< Tanh layer
	nnl2_nn_type_relu,			///< ReLU layer
	nnl2_nn_type_leaky_relu,	///< Leaky-ReLU layer
    nnl2_nn_type_unknown        ///< Unknown or unsupported network type 
} nnl2_nn_type;

///@} [nnl2_nn_type]

#endif /** NNL2_NN_TYPE_DEFINED **/



#ifndef NNL2_NNLREPR_TEMPLATE_DEFINED
#define NNL2_NNLREPR_TEMPLATE_DEFINED

///@{ [nnl2_nnlrepl_template]

/** @struct nnl2_nnlrepr_template_item
 ** @brief 
 * nnlrepr is a nnl2 nn own data format
 * for representing neural network as a
 * 1D tensor and back. Needs for GA 
 * algorithms
 */
typedef struct nnl2_nnlrepr_template_item {
    nnl2_nn_type nn_type;		///< Type of the neural network layer
	nnl2_tensor_type dtype;		///< Data type used for tensors in current layer
	int** shapes; 				///< Array of shape definitions for the layer
    size_t vector_size;         ///< Encoded 1d vector size
    size_t num_shapes;          ///< Number of shape definitions
    size_t num_childrens;       ///< Number of child layers in this template
	void* additional_data;      ///< Additional layer-specific data
    struct nnl2_nnlrepr_template_item** childrens;      ///< Array of child layer templates
} nnl2_nnlrepr_template;

///@} [nnl2_nnlrepl_template]

#endif /** NNL2_NNLREPR_TEMPLATE_DEFINED **/


///@{ [nnl2_nn_handle_as]

/** @enum nnl2_nn_handle_as
 ** @brief 
 * Enums for processing manual tensor transfers in neural networks
 * This means that you can pass your own tensors to fnn, for example, 
 * to initialize them instead of writing :init :kaiming/uniform or 
 * something similar
 */
typedef enum {
	nnl2_nn_handle_as_copy = 0,  ///< Make a copy of the passed tensors
    nnl2_nn_handle_as_view = 1   ///< Take tensor pointers and work with them directly
} nnl2_nn_handle_as;

///@} [nnl2_nn_handle_as]



///@{ [nnl2_nn_ann]

typedef struct {
    nnl2_nn_type nn_type;  ///< Type of the neural network 
	uint32_t nn_magic; 	   ///< To check if the pointer is a nn structure
    bool use_bias;         ///< Whether the network uses bias terms 
} nnl2_nn_ann;

///@} [nnl2_nn_ann]



/** @brief Magic number to check if the pointer is a nn structure (nnl2_nn_ann->nn_magic) **/
#define NNL2_NN_MAGIC 0XABCDFDCB

/** @brief If ann returns true false otherwise **/
bool nnl2_nn_checktype(void* obj) {
	bool result;
	
    if(!obj) result = false;
    
    nnl2_nn_ann* ann = (nnl2_nn_ann*)obj;
    
    if(ann->nn_magic != NNL2_NN_MAGIC) result = false;
    
    if(ann->nn_type >= nnl2_nn_type_fnn && ann->nn_type <= nnl2_nn_type_unknown) {
        result = true;
    }
	
	return result;
}

/** @brief 
 * Retrieves the type of a neural network
 *
 ** @param nn 
 * Pointer to the neural network structure
 *
 ** @return nnl2_nn_type
 * The type of the neural network, or nnl2_nn_type_unknown on error
 *
 ** @see nnl2_nn_type
 **/
nnl2_nn_type nnl2_nn_get_type(void* nn) {
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
    #endif
    
    #if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
        if (nn == NULL) {
            #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_MINIMAL
                NNL2_DEBUG("In function nnl2_nn_get_type, nn is NULL. returning unknown type (nnl2_nn_type_unknown)");
            #endif
                
            return nnl2_nn_type_unknown;
        }
    #endif
    
    nnl2_nn_type* nn_type_mem = (nnl2_nn_type*)nn;
    nnl2_nn_type result = *nn_type_mem;
    
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_EXIT();
    #endif
    
    return result;
}

#endif /** NNL2_ANN_BACKEND_H **/