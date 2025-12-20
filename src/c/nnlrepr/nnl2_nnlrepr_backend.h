#ifndef NNL2_NNLREPR_BACKEND_H
#define NNL2_NNLREPR_BACKEND_H

/** @file nnl2_nnlrepr_backend.h 
 ** @copyright MIT License
 ** @date 2025
 ** @brief nnlrepr encoder, decoder backend
 **
 ** Filepath: src/c/nnlrepr/nnl2_nnlrepr_backend.h
 **
 ** In case of errors/problems/suggestions, please write to issues or nnl.dev@proton.me
 ** nnl2 Repository: https://github.com/danish-song-of-liberation/nnl2
 **/

#ifndef NNL2_NN_TYPE_DEFINED
#define NNL2_NN_TYPE_DEFINED

///@{ [nnl2_nn_type]

typedef enum {
    nnl2_nn_type_fnn 		 =  0,    ///< Fully Connected Neural Network 
	nnl2_nn_type_rnn_cell    =  1,    ///< Vanilla Recurrent Neural Network Cell
	nnl2_nn_type_rnn     	 =  2,    ///< Vanilla Recurrent Neural Network
	nnl2_nn_type_sequential  =  3,    ///< Sequential neural network (layers in sequence)
	nnl2_nn_type_sigmoid     =  4,	  ///< Sigmoid layer
	nnl2_nn_type_tanh 		 =  5,	  ///< Tanh layer
	nnl2_nn_type_relu 		 =  6,	  ///< ReLU layer
	nnl2_nn_type_leaky_relu  =  7,	  ///< Leaky-ReLU layer
    nnl2_nn_type_unknown     =  8     ///< Unknown or unsupported network type 
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



/** @brief Encodes by nnlrepr format passed neural network **/
nnl2_nnlrepr_template* nnl2_ann_nnlrepr_template(void* nn);

/** @brief Retrieves all parameters from an ANN model **/
nnl2_ad_tensor** nnl2_ann_parameters(void* nn);

/** @brief Returns the number of parameters in an ANN model **/
size_t nnl2_ann_num_parameters(void* nn);



///@{ [nnl2_nnlrepr]

/** @struct nnl2_nnlrepr
 ** @brief Full nnlrepr structure
 **/
typedef struct {
    nnl2_ad_tensor* vector;   		     ///< 1D tensor containing serialized network data
    nnl2_nnlrepr_template* template;     ///< Template describing network architecture and hierarchy
} nnl2_nnlrepr;

///@} [nnl2_nnlrepr]



nnl2_nnlrepr* nnl2_nn_encode(void* nn);

/** @brief 
 * Safely frees memory allocated for nnlrepr template structure
 * 
 ** @param tpl (template)
 * Pointer to template structure to free (can be NULL)
 *
 ** @warning
 * I've spent 5+ hours to fix UB here
 *
 ** @warning 
 * Black magic
 *
 ** @warning 
 * I do not know why it works
 *
 ** @note
 * UPD: I figured out why it wasn't working
 */
void nnl2_nnlrepr_template_free(nnl2_nnlrepr_template* tpl) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_ENTER();
		NNL2_INFO("Freeing %p nnlrepr template", tpl);
	#endif
	
	if(tpl == NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
            NNL2_INFO("nnl2_nnlrepr_template_free: NULL pointer, skipping");
            NNL2_FUNC_EXIT();
        #endif
		
        return;
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("nnl2_nnlrepr_template_free: freeing template %p (nn_type: %d, type: %d, num_childrens: %zu, num_shapes: %zu)", 
                  tpl, tpl->nn_type, tpl->nn_type, tpl->num_childrens, tpl->num_shapes);
    #endif
	
	// Recursive freeing 
	if(tpl->childrens != NULL && tpl->num_childrens > 0) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("nnl2_nnlrepr_template_free: freeing %zu children of template %p", tpl->num_childrens, tpl);
        #endif
		
        for(size_t i = 0; i < tpl->num_childrens; i++) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
                NNL2_INFO("nnl2_nnlrepr_template_free: processing child %zu/%zu at %p", 
                          i + 1, tpl->num_childrens, tpl->childrens[i]);
            #endif
			
            if(tpl->childrens[i] != NULL) {
                nnl2_nnlrepr_template_free(tpl->childrens[i]);
				tpl->childrens[i] = NULL;
            }
        }
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("nnl2_nnlrepr_template_free: freeing childrens array at %p", tpl->childrens);
        #endif
		
        free(tpl->childrens);
		
		tpl->childrens = NULL;
		tpl->num_childrens = 0;
    }
    
	// Free shapes
    if(tpl->shapes != NULL && tpl->num_shapes > 0) {
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("nnl2_nnlrepr_template_free: freeing %d shapes of template %p", tpl->num_shapes, tpl);
        #endif
		
		/* // Already freed
		   // Double freeing 
		   // UB
        for(size_t i = 0; i < tpl->num_shapes; i++) {
			#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
				NNL2_INFO("nnl2_nnlrepr_template_free: freeing shape %d/%d at %p", i + 1, tpl->num_shapes, tpl->shapes[i]);
			#endif 
			
            if(tpl->shapes[i] != NULL) {
				#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
					NNL2_INFO("nnl2_nnlrepr_template_free: shapes[%zu][0]: %d", i, tpl -> shapes[i][0]);
					NNL2_INFO("nnl2_nnlrepr_template_free: shapes[%zu][1]: %d", i, tpl -> shapes[i][1]);
				#endif
			
                //free(tpl->shapes[i]);  // Skip it
                //tpl->shapes[i] = NULL; // Skip it
            }
        }
		*/
		
		#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("nnl2_nnlrepr_template_free: freeing shapes array at %p", tpl->shapes);
        #endif
		
        free(tpl->shapes);
        tpl->shapes = NULL;
		tpl->num_shapes = 0;
    }
    
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("nnl2_nnlrepr_template_free: freeing template structure at %p", tpl);
    #endif
    
    free(tpl);
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_FUNC_EXIT();
	#endif
	
	return;
}

void nnl2_nn_print_encoder(nnl2_nnlrepr_template* encoder, bool terpri, int depth);

/** @brief 
 * Frees all memory associated with an nnlrepr structure
 * 
 ** @param nnlrepr 
 * Pointer to nnlrepr structure to free
 */
void nnl2_nnlrepr_free(nnl2_nnlrepr* nnlrepr) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_FUNC_ENTER();
        NNL2_INFO("(nnlrepr) Freeing nnlrepr at %p", nnlrepr);
    #endif
	
	/* see nnl2_nnlrepr_template_free doxygen
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
		NNL2_INFO("nnl2_nnlrepr_free: shapes[0][0]: %d", nnlrepr -> template -> shapes[1][0]);
		NNL2_INFO("nnl2_nnlrepr_free: shapes[0][1]: %d", nnlrepr -> template -> shapes[1][1]);
		NNL2_INFO("nnl2_nnlrepr_free: shapes[1][0]: %d", nnlrepr -> template -> shapes[2][0]);
	#endif
	*/
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
		if(nnlrepr == NULL) {
			NNL2_ERROR("In function nnl2_nnlrepr_free, nnlrepr is NULL. Early return");
			return;
		}
	#endif 
	
	// Vector freeing
	if(nnlrepr -> vector != NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("(nnlrepr) Freeing nnlrepr vector at %p", nnlrepr -> vector);
        #endif
		
        nnl2_free_ad_tensor(nnlrepr -> vector);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			nnlrepr -> vector = NULL;
		#endif
    }
	
	// Template freeing
	if(nnlrepr -> template != NULL) {
        #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
            NNL2_INFO("(nnlrepr) Freeing nnlrepr template at %p", nnlrepr -> template);
        #endif
		
        nnl2_nnlrepr_template_free(nnlrepr->template);
		
		#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN
			nnlrepr -> template = NULL;
		#endif
    }
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Freeing nnlrepr structure at %p", nnlrepr);
    #endif
	
	// Structure freeing
    free(nnlrepr);
	
    #if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_VERBOSE
        NNL2_INFO("Successfully freed nnlrepr");
        NNL2_FUNC_EXIT();
    #endif
	
	return;
}

nnl2_nnlrepr* nnl2_nnlrepr_empty_from_vector(nnl2_ad_tensor* self_vector) {
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_ENTER();
    #endif
	
	nnl2_nnlrepr* encoder = malloc(sizeof(nnl2_nnlrepr));
	
	#if NNL2_SAFETY_MODE >= NNL2_SAFETY_MODE_MIN 
		if(encoder == NULL) {
			NNL2_MALLOC_ERROR();
			return NULL;
		}
	#endif 
	
	encoder -> vector = self_vector;
	encoder -> template = NULL;
	
	#if NNL2_DEBUG_MODE >= NNL2_DEBUG_MODE_FULL
        NNL2_FUNC_EXIT();
    #endif
	
	return encoder;
}

#endif /** NNL2_NNLREPR_BACKEND_H **/
